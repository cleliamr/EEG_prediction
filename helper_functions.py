import numpy as np 
import pandas as pd
from mne import compute_covariance, cov, read_epochs, filter
from scipy.signal import lfilter
import gc

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