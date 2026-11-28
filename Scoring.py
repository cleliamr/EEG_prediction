import numpy as np
import pandas as pd
from mne import compute_covariance, cov, read_epochs, filter
from scipy.signal import lfilter
import gc
import os
##
import xarray as xr
from sklearn.metrics import (
    r2_score,
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
)



"""Scoring synthetic EEG (from a fitted HMP model) against real EEG.

Main entry point: `score_all(epoch_data, simulated, rt_samples)`.
Both inputs are xarray Datasets with a 'data' variable and dims
(recording(=participant), epoch(=trial), channel(=electrode), sample), as returned by
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
    with np.errstate(invalid="ignore", divide="ignore"):  # ignoring warnings
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
              agg="mean", min_trial_frac=0.5):
    """Average the signals as requested, then compute `metric` per electrode.

    Parameters
    ----------
    real_ds, pred_ds : xarray Datasets, dims (recording, epoch, channel, sample)
    metric : "r2", "explained_variance", "mse", "mae" or "corr"
    average_trial : True -> average the signals over trials before scoring
    average_participant : True -> average the signals over participants before scoring
    average_channel : True -> average the electrode SCORES at the end
    agg : "mean" or "median", used only to average the electrode scores
    min_trial_frac: cutoff for grand average to account for fact that all trials don't have the same length.
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

    # Averaging over trials: keep only the samples where at least `min_trial_frac`
    # of the scored trials are still running (they stop at their RT)
    if (average_trial or average_participant) and min_trial_frac:
        running = valid.any("channel").sum(["recording", "epoch"])  # trials left at each sample
        n_total = valid.any(["channel", "sample"]).sum(["recording", "epoch"])
        keep = running >= min_trial_frac * n_total  # (sample,)
        real, pred = real.where(keep), pred.where(keep)

    # 1. Average the signals
    with np.errstate(invalid="ignore"):  # ignore warning (samples after every RT -> empty mean)
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
    R = real.values.reshape(-1, n_ch, n_s)  # (n_blocks, channel, sample)
    P = pred.values.reshape(-1, n_ch, n_s)
    out = np.full((R.shape[0], n_ch), np.nan)
    for b in range(R.shape[0]):  # b = block number: 0, 1, 2, ...
        # block = piece of EEG of shape (electrodes, time) that gets scored separately
        # Nothing averaged: One block = One trial of one participant
        # average_trial=True: One block = The ERP of one participan
        # average_trial + average_participant: One block = The grand-average ERP
        if np.isnan(R[b]).all():  # empty block (rejected trial) → skip
            continue
        out[b] = per_channel(R[b], P[b], metric)  # returns 32 scores, one per electrode, stored in row b of out

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


# keep only the samples where at least `min_trial_frac` of the scored trials are still running
def trial_cutoff(real_ds, min_frac=0.5, trial_mask=None):
    """Last sample at which at least `min_frac` of the trials are still running."""
    ok = real_ds["data"].notnull().all("channel")                  # (recording, epoch, sample)
    if trial_mask is not None:
        ok = ok & trial_mask
    running = ok.sum(["recording", "epoch"])
    n_total = ok.any("sample").sum(["recording", "epoch"])
    return int(running.sample.where(running >= min_frac * n_total).max())

# Test/Train split

def split_trials(epoch_data, test_frac=0.2, seed=0, by="trial"):
    """Choose which trials are held out for testing.

    Parameters
    ----------
    test_frac : fraction of trials (or participants) used for testing.
    seed      : random seed, so the split is reproducible.
    by        : "trial"       -> random trials, mixed across participants.
                                 Easy, but trials of one participant are
                                 correlated, so the score is a bit optimistic.
                "participant" -> whole participants are held out.
                                 Stricter (tests generalisation to new people),
                                 but needs several participants.

    Returns
    -------
    Boolean DataArray (recording, epoch): True = test trial, False = train trial.
    Cells that hold no data (rejected trials) are also assigned; they are NaN
    and therefore ignored later.
    """
    rng = np.random.default_rng(seed)
    n_rec = epoch_data.sizes["recording"]
    n_ep = epoch_data.sizes["epoch"]

    if by == "participant":
        n_test = max(1, int(round(test_frac * n_rec)))
        test_rec = rng.choice(n_rec, n_test, replace=False)
        m = np.zeros((n_rec, n_ep), dtype=bool)
        m[test_rec, :] = True
    elif by == "trial":
        m = rng.random((n_rec, n_ep)) < test_frac
    else:
        raise ValueError("by must be 'trial' or 'participant'")

    return xr.DataArray(
        m,
        dims=("recording", "epoch"),
        coords={"recording": epoch_data.recording, "epoch": epoch_data.epoch},
    )


"""Score the prediction around each event (window scoring).

THE IDEA
--------
For each event, keep only the samples in a short window around the event time
(in every trial) and score the prediction on those samples:

    "How well is the EEG reproduced where this event happens?"

Three things are needed: the real EEG, the prediction (from any model, in
the same unit as the real EEG) and the event times.

By default (`pooled=True`) the windows of all selected trials are pooled into one
long signal per electrode before scoring. A window is only ~13 samples long, so
scoring it in every single trial would be very noisy.

NOTE: R2 is centred on the window, i.e. the baseline is the mean of the window
samples. The score therefore measures how well the shape and amplitude inside
the window are reproduced, and it is not comparable with a score on full trials.
"""


# =============================================================================
# 1. Score with all trials pooled into one signal per electrode
# =============================================================================
def pooled_score(real_ds, pred_ds, metric="r2"):
    """Score with all trials pooled into one long signal per electrode.

    Every valid (trial, sample) point is stacked into one column per electrode
    and `per_channel` is applied to it. NaN points (rejected trials, samples
    after the RT, samples outside a window) are ignored.

    Returns a DataArray (channel,), NaN if fewer than 2 points are valid.
    """
    dims = ("channel", "recording", "epoch", "sample")
    y = real_ds["data"].transpose(*dims)
    p = pred_ds["data"].transpose(*dims)
    n_ch = y.sizes["channel"]
    Y = y.values.reshape(n_ch, -1)  # (channel, all points of all trials)
    P = p.values.reshape(n_ch, -1)

    ok = ~(np.isnan(Y).any(axis=0) | np.isnan(P).any(axis=0))
    if ok.sum() < 2:
        out = np.full(n_ch, np.nan)
    else:
        out = per_channel(Y, P, metric)
    return xr.DataArray(out, dims="channel", coords={"channel": y.channel.values})


# =============================================================================
# 2. The window around each event
# =============================================================================
def event_windows(real_ds, times, half_width):
    """Boolean mask of the samples around each event, for every trial.

    For trial i and event k the window is
        [event_time - half_width, event_time + half_width]
    clipped to the trial (not before sample 0, not from the RT onwards).

    Parameters
    ----------
    real_ds    : Dataset with dims (recording, epoch, channel, sample); only its
                 coordinates and sizes are used.
    times      : DataArray from hmp.utils.event_times(..., add_rt=True): one row
                 per trial, one column per event, the LAST column is the RT.
                 It must carry `recording` and `epoch` coordinates on its trial axis.
    half_width : half the window, in samples (e.g. 6 -> 13 samples = 52 ms at
                 250 Hz, about the width of a 50 ms half-sine).

    Returns
    -------
    Boolean DataArray (event, recording, epoch, sample).
    """
    other = [d for d in times.dims if d != "trial"][0]
    t = times.transpose("trial", other).values  # (trial, n_events + 1)
    n_events = t.shape[1] - 1  # last column = RT

    rec_idx = {str(r): i for i, r in enumerate(real_ds.recording.values)}
    ep_idx = {int(e): i for i, e in enumerate(real_ds.epoch.values)}
    n_rec, n_ep, n_s = (real_ds.sizes[d] for d in ("recording", "epoch", "sample"))

    masks = np.zeros((n_events, n_rec, n_ep, n_s), dtype=bool)
    for i, (r, e) in enumerate(zip(times.recording.values, times.epoch.values)):
        ri, ei = rec_idx[str(r)], ep_idx[int(e)]
        rt = int(t[i, -1])
        for k in range(n_events):
            c = int(t[i, k])
            lo, hi = max(c - half_width, 0), min(c + half_width + 1, rt)
            if hi > lo:
                masks[k, ri, ei, lo:hi] = True

    return xr.DataArray(
        masks, dims=("event", "recording", "epoch", "sample"),
        coords={"event": np.arange(n_events), "recording": real_ds.recording,
                "epoch": real_ds.epoch, "sample": real_ds.sample},
    )


# =============================================================================
# 3. Score each event
# =============================================================================
def score_event_windows(real_ds, pred_ds, times, half_width, metric="r2",
                        trial_mask=None, pooled=True, **score_kwargs):
    """Score the prediction only around each event.

    Parameters
    ----------
    real_ds, pred_ds : Datasets (recording, epoch, channel, sample), real EEG and
                 prediction, in the same unit.
    times      : event times, last column = RT (see `event_windows`).
    half_width : half the window, in samples.
    metric     : "r2", "explained_variance", "mse", "mae" or "corr".
    trial_mask : optional boolean DataArray (recording, epoch); only the trials
                 where it is True are scored (e.g. the test trials).
    pooled     : True -> one value per event and electrode, trials pooled.
                 False -> use `score(...)`; extra keyword arguments such as
                 average_trial=True are passed on to it.

    Returns
    -------
    DataArray (event, channel) when pooled=True.
    """
    dims = ("recording", "epoch", "channel", "sample")
    real = real_ds["data"].transpose(*dims)
    pred = pred_ds["data"].transpose(*dims)
    if trial_mask is not None:
        real, pred = real.where(trial_mask), pred.where(trial_mask)

    windows = event_windows(real_ds, times, half_width)
    results = []
    for k in windows.event.values:
        m = windows.sel(event=k)  # (recording, epoch, sample)
        r = xr.Dataset({"data": real.where(m)})
        p = xr.Dataset({"data": pred.where(m)})
        results.append(pooled_score(r, p, metric) if pooled
                       else score(r, p, metric, **score_kwargs))
    return xr.concat(results, dim="event").assign_coords(event=windows.event.values)