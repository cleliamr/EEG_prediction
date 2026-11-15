import numpy as np 
import pandas as pd
from mne import compute_covariance, cov, read_epochs, filter
from scipy.signal import lfilter
import gc
##
import xarray as xr
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
)

# Defininf a half-sine, as the one expected by HMP
class HalfSine:
    def __init__(self, sfreq, width):
        steps = 1000/sfreq
        width_samples = width/steps

        # time (ms) of each sample of the bump, taken at the middle of each sample
        event_idx = np.arange(width_samples) * steps + steps / 2
        # gives event frequency given that events are defined as half-sines
        event_frequency = 1000 / (width * 2)
        
        template = np.sin(2 * np.pi * event_idx / 1000 * event_frequency)
        self.template = template

# Class for evidence accumulation traces
class EvidenceAccumulation:
    def __init__(self, sfreq, drift_rate, time_steps=3000, n_simulations=10000):
        traces = np.zeros((n_simulations, time_steps))
        decision_times = np.zeros(n_simulations)
        #Initiate a bunch of traces
        for i in range(n_simulations):
            trace, decision_time = self.accumulation_trace(sfreq, drift_rate, time_steps)
            traces[i, :len(trace)] = trace
            decision_times[i] = decision_time
        self.traces = traces
        self.decision_times = decision_times

    def accumulation_trace(self, sfreq, drift_rate, time_steps):
        noise_std = .1 
        threshold = 1 
        #Random walk
        evidence = np.cumsum(np.random.normal(drift_rate, noise_std, time_steps))
        decision_time = np.where(np.abs(evidence) >= threshold)[0]
        if len(decision_time) > 0:
            decision_time = decision_time[0]
        else:
            decision_time = time_steps
        # Invert trace if threshold is neg
        if evidence[decision_time] <= -threshold:
            evidence = -evidence
        return evidence[:decision_time+1], decision_time
    
    def find_closest(self, time):
        # Find the trace with the closest time
        closest_idx = np.argmin(np.abs(self.decision_times - time))
        trace = self.traces[closest_idx]
        return trace[:int(time)+1]

def simulate_from_hmpfit(epoch_data,
                         weights,
                         times,
                         template_dict,
                         add_noise=False
                        ):
    sfreq = epoch_data.sfreq
    n_chan, _, n_events = np.shape(weights)
    surrogate = epoch_data.copy(deep=True)
    surrogate['data'] = 0 * surrogate['data'] #Overwrite all the real data
    surrogate = surrogate.stack({'trial':["recording","epoch"]})

    # Add event activations at corresponding times
    for trial in times.trial:
        #for each trial: get events times and RT
        trial_times = times.sel(trial=trial).values
        trial_rt = int(trial_times[-1])

        # use weights of each event, sum over events
        for event in range(n_events):
            trial_template = template_dict[event]
            
            # EA traces are defined from 2nd event to last
            # 2 possibilities:
            # 1) template: evidence accumulation (the ramp)
            if isinstance(trial_template, EvidenceAccumulation):
                prev_ev = trial_times[event-2].astype(int)
                dist_from_prev = trial_times[event]-prev_ev
                trial_template = trial_template.find_closest(dist_from_prev)
                length_of_event = len(trial_template)
                proposed_onset = prev_ev
                proposed_offset = prev_ev+length_of_event
            # 2) template: half-sine
            else:# Half-Sine's peak is symetric
                trial_template = trial_template.template
                length_of_event = len(trial_template)
                # centered bump: starts half a bump before and ends half a bump after.
                proposed_onset = trial_times[event].astype(int) - length_of_event // 2
                proposed_offset = trial_times[event].astype(int) + length_of_event // 2 + 1 # + 1 because last index not taken

                # if length_of_event even, it would give one sample too many, so the if removes it.
                if length_of_event%2 == 0:
                    proposed_offset -= 1

            # Ensures that pattern doesn't exceed stim/RT boundaries if detected event close to those
            # If an event is very close to the stimulus (time 0) or to the response (RT), part of the bump would fall outside the trial.
            onset = max(proposed_onset, 0) # The bump can’t start before 0. If it would, start at 0 instead
            offset = min(proposed_offset, trial_rt) # The bump can’t end after the RT. If it would, end at the RT instead
            length = offset - onset
            trim_start = onset - proposed_onset # How many template values to drop at the start
            trim_end = proposed_offset - offset # How many template values to drop at the end

            # Every bump value × every electrode weight of event into the empty EEG
            surrogate.sel(trial=trial)['data'][:,onset:offset] += \
                np.outer(trial_template[trim_start:length_of_event - trim_end],\
                         weights.sel(trial = trial, event=event).values).T

    surrogate = surrogate.unstack().transpose('recording','epoch', 'channel', 'sample')

    if add_noise:
        # Add noise, using recording covariance among electrodes and an IIR filter
        for recording in surrogate.recording:
            surrogate.sel(recording=recording)["data"] += create_noise(recording.values, \
                        sfreq, n_trials=len(surrogate.epoch), n_samples=np.shape(surrogate.sel(recording=recording)["data"])[2])
   
    surrogate = surrogate.unstack()
    return surrogate

def create_noise(recording, sfreq, n_trials, n_samples):
    """Create spatially colored and temporally IIR-filtered noise.
    
    Adapted from: https://github.com/mne-tools/mne-python/blob/maint/1.9/mne/simulation/evoked.py#L171
    """

    # Use actual epoched data = load the participant’s real EEG
    epochs = read_epochs(f'preprocessed/{recording}.fif', verbose=False)
    epochs = epochs.pick('eeg')
    epochs = epochs.resample(100)#Speed

    # measure the real noise
    # how electrodes vary together, measured only on the 200 ms before the stimulus
    data_cov = compute_covariance(epochs, keep_sample_mean=False, tmin=-.2,tmax=0, verbose=False)#Use only baseline, less actual ERPs
    # typical amplitude of the noise, from the first 20 samples
    data_std = np.std(epochs.get_data()[:,:,:20])

    # whitener takes correlated noise and makes it independent (removes the correlations)
    data_cov = cov.prepare_noise_cov(data_cov, epochs.info, verbose=False)
    # colorer does the reverse: takes independent noise and adds the correlations of the real data.
    _, _, colorer = cov.compute_whitener(data_cov, pca=True, return_colorer=True, verbose=False)

    # generate noise for each trial
    noise_matrix = np.zeros((n_trials, len(epochs.info["ch_names"]), n_samples))
    for trial in range(n_trials):
        # A) spatial correlation
        # takes pure random numbers, independent on each electrode and sample
        # and apply colorer: noise now correlated across electrodes like real EEG.
        noise = np.dot(colorer, np.random.standard_normal((colorer.shape[1], n_samples)))

        # B) temporal correlation (the IIR filter)
        # Each new noise value depends on the previous two, so the noise becomes smooth over time, like real EEG, instead of jumping randomly.
        zi = np.zeros((len(colorer), len([0.2, -0.2, 0.04]) - 1))
        noise, _ = lfilter([1], [0.2, -0.2, 0.04], noise, axis=-1, zi=zi) # axis=-1 = filter along time, zi = start the filter from zero

        # C) right amplitude
        # Rescale so the noise has the same standard deviation as the real baseline
        noise = noise * (data_std / np.std(noise))
        noise_matrix[trial] = noise
    return noise_matrix


###### NEW CODE ######
"""Scoring synthetic EEG (from a fitted HMP model) against real EEG.

Main entry point: `score_all(epoch_data, simulated, rt_samples)`.
Both inputs are xarray Datasets with a 'data' variable and dims
(recording, epoch, channel, sample), as returned by
`hmp.io.read_mne_epochs` and `simulate_from_hmpfit`.

Metrics use sklearn.metrics where a function exists
(r2_score, explained_variance_score, mean_squared_error, mean_absolute_error).
sklearn has no Pearson correlation, so that one is computed with numpy.
"""
# ====================================================================================
# 1. METRICS FOR ONE BLOCK OF EEG (channel x sample) -> ONE VALUE PER ELECTRODE
# ====================================================================================
def pearson(y_true, y_pred, multioutput="raw_values"):
    """Pearson correlation per column, with the same signature as sklearn metrics.

    # sklearn's `multioutput` option: what to return when there are several outputs
    # (here, several electrodes = several columns)
    #
    #   "uniform_average"    (default)  -> plain average of the electrodes' scores: one single number
    #   "variance_weighted"             -> average weighted by each electrode's variance: one number
    #                                      (only for r2_score and explained_variance_score)
    #   "raw_values"                    -> one score per electrode: array of shape (n_electrodes,)
    #
    # Use "raw_values" so the score is computed per electrode; averaging over
    # electrodes is done afterwards, only if requested (by_channel=False in aggregate).

    y_true, y_pred : arrays of shape (n_samples, n_channels).
    Returns an array of shape (n_channels,).
    """
    yt = y_true - y_true.mean(axis=0)
    yp = y_pred - y_pred.mean(axis=0)
    den = np.sqrt((yt ** 2).sum(axis=0) * (yp ** 2).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"): # ignoring warnings
        return (yt * yp).sum(axis=0) / den  # flat signal -> NaN


METRICS = {
    "r2": r2_score,
    "explained_variance": explained_variance_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
    "corr": pearson,
}


def per_channel(y, yhat, metric):
    """Apply a metric to one block of EEG, one value per electrode.

    y, yhat : arrays of shape (channel, sample) (one trial, or one ERP).

    sklearn constraints handled here:
    - it expects (n_samples, n_outputs): time points are the samples and
      electrodes the outputs, so the block is transposed;
    - it crashes on NaN: only time points valid on every electrode are kept
      (within a trial, all electrodes share the same valid samples, i.e. up to the RT).

    Returns an array of shape (channel,), NaN if fewer than 2 valid samples.
    """
    keep = ~(np.isnan(y).any(axis=0) | np.isnan(yhat).any(axis=0))
    # `multioutput="raw_values"` returns one score per electrode, not their mean
    return METRICS[metric](y[:, keep].T, yhat[:, keep].T, multioutput="raw_values")


# -----------------------------------------------------------------------------
# 2. Averaging the signals
# -----------------------------------------------------------------------------
def average_signals(da, average_trial=False, average_participant=False):
    """Average an EEG DataArray (recording, epoch, channel, sample) over trials
    and/or participants. NaNs (rejected trials, samples after the RT) are skipped."""
    if average_participant and not average_trial:
        print("average_participant=True requires average_trial=True: average_trial was set to True")
        print("(trials of different participants can't be averaged together)")
        average_trial = True
    if average_trial:
        da = da.mean("epoch", skipna=True)  # 1. ERP of each participant
    if average_participant:
        da = da.mean("recording", skipna=True)  # 2. average of the ERPs
    return da



# -----------------------------------------------------------------------------
# 3. Score
# -----------------------------------------------------------------------------
def score(real_ds, pred_ds, metric="r2",
          average_trial=False, average_participant=False, average_channel=False,
          agg="mean"):
    """Average the signals as requested, then compute `metric` per electrode.

    Parameters
    ----------
    real_ds, pred_ds : xarray Datasets, dims (recording, epoch, channel, sample)
    metric : "r2", "explained_variance", "mse", "mae" or "corr"
    average_trial : True -> average the signals over trials before scoring
    average_participant : True -> average the signals over participants before scoring
    average_channel : True -> average the electrode SCORES at the end
    agg : "mean" or "median", used only to average the electrode scores

    Returns
    -------
    xarray DataArray with the kept dims (see the table at the top of the file).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {list(METRICS)}")

    dims = ("recording", "epoch", "channel", "sample")
    real = real_ds["data"].transpose(*dims).astype(float)
    pred = pred_ds["data"].transpose(*dims).astype(float)

    # Same NaNs in both, BEFORE averaging: both averages then use exactly the same trials
    valid = real.notnull() & pred.notnull()
    real, pred = real.where(valid), pred.where(valid)

    # 1. Average the signals
    with np.errstate(invalid="ignore"):              # ignore warning (samples after every RT -> empty mean)
        real = average_signals(real, average_trial, average_participant)
        pred = average_signals(pred, average_trial, average_participant)

    # 2. Score per electrode, for every remaining block (trial / participant / grand average)
    # After average_signals, the dimensions left depend on what averaged:
    # Nothing averaged	(recording, epoch, channel, sample)
    # average_trial=True	(recording, channel, sample)
    # average_trial + average_participant	(channel, sample)
    lead = [d for d in real.dims if d not in ("channel", "sample")]
    # Puts channel and sample at the end, the others in front
    real = real.transpose(*lead, "channel", "sample")
    pred = pred.transpose(*lead, "channel", "sample")
    # Number of electrodes and time points
    n_ch, n_s = real.sizes["channel"], real.sizes["sample"]
    lead_shape = tuple(real.sizes[d] for d in lead)

    # All dimensions  merged into one
    R = real.values.reshape(-1, n_ch, n_s)           # (n_blocks, channel, sample)
    P = pred.values.reshape(-1, n_ch, n_s)
    out = np.full((R.shape[0], n_ch), np.nan)
    for b in range(R.shape[0]):                     # b = block number: 0, 1, 2, ...
        # block = piece of EEG of shape (electrodes, time) that gets scored separately
        # Nothing averaged: One block = One trial of one participant
        # average_trial=True: One block = The ERP of one participan
        # average_trial + average_participant: One block = The grand-average ERP
        if np.isnan(R[b]).all():                     # empty block (rejected trial) → skip
            continue
        out[b] = per_channel(R[b], P[b], metric)    # returns 32 scores, one per electrode, stored in row b of out

    coords = {d: real[d].values for d in lead}
    coords["channel"] = real.channel.values
    scores = xr.DataArray(out.reshape(*lead_shape, n_ch),
                          dims=(*lead, "channel"), coords=coords)

    # 3. Average the electrode scores (not the signals):
    if average_channel:
        if agg == "median":
            scores = scores.median("channel", skipna=True)
        else:
            scores = scores.mean("channel", skipna=True)

    return scores