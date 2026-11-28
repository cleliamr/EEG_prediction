"""Least-squares fit of shared event weights for HMP-based EEG prediction.

HMP says that the EEG of one trial is a sum of short "bumps" (half-sines), one
per event, each placed at the event's estimated time. Each bump is spread over
the electrodes with a scalp map (the "weights" of the event):

    EEG[channel c, time t]  ~  sum over events e of  W[e, c] * T_e(t)

    T_e(t) : the bump of event e, placed at that trial's event time
             (0 everywhere else). It is the same for all electrodes.
    W[e, c]: how strongly event e shows up on electrode c.

In `simulate_from_hmpfit` the weights are read from each trial's own data
(one scalp map per trial and event). That is not a real prediction, because
the weights already "saw" the trial they are supposed to predict.

Here we instead try to learn one weight per (event, electrode), shared by all trials:

    1. Split the trials into a training set and a test set.
    2. Build T_e(t) for every trial (reusing the simulation code).
    3. Find W that minimises the squared error on the training trials.
       The model is linear in W, so this is ordinary least squares and has an
       exact solution.
    4. Predict the EEG of all trials with that W.
    5. Score the prediction on the held-out test trials.

EXPECTED INPUT FORMAT
---------------------
epoch_data : xarray Dataset, variable 'data', dims
             (recording, epoch, channel, sample). Trials that HMP did not use,
             and samples after each trial's RT, must already be NaN
             (steps 4 and 5 of the loading code).
weights    : DataArray (channel, trial, event), from hmp.utils.event_channels.
             Only its shape is used here; the values are overwritten.
times      : DataArray (trial, event+1), from hmp.utils.event_times(add_rt=True).
template_dict : {event index: HalfSine(...)} as used in simulate_from_hmpfit.

Units: EEG is stored in volts (values ~1e-6). We convert to microvolts so the
weights are readable. R2 does not change when the data is rescaled.
"""

import numpy as np
import xarray as xr

from helper_functions import simulate_from_hmpfit
from Scoring import split_trials

UV = 1e6  # volts -> microvolts



# Build the bump of each event, for every trial
def event_traces(epoch_data, weights, times, template_dict):
    """Return T_e(t) for every event and every trial.

    How: `simulate_from_hmpfit` already knows how to place a half-sine at an
    event time, including the clipping at sample 0 and at the RT. We call it
    once per event with weights equal to 1 on electrode 0 and 0 everywhere
    else, so that the simulated signal on electrode 0 IS the bump T_e(t).
    Reusing the function guarantees the bumps are built exactly like in the
    simulation (same placement, same clipping).

    Cost: one simulation run per event (it loops over all trials in Python).

    Returns
    -------
    DataArray with dims (event, recording, epoch, sample). It is NaN wherever
    epoch_data is NaN (unused trials, samples after the RT).
    """
    weights = weights.transpose("channel", "trial", "event")
    traces = []
    for e in range(weights.sizes["event"]):
        w = xr.zeros_like(weights)
        w.values[0, :, e] = 1.0          # electrode 0, all trials, event e only -> at end you get a matrice with 1 alternatively for each event
                                            # T_e(t) is the same on every electrode/ all the electrodes have same events times so only done for one electrode
        sim = simulate_from_hmpfit(epoch_data, w, times, template_dict)  # dims (recording = nbr of subjects, epoch = trials, channel = electrodes, sample = length of epoch)
        traces.append(
            sim["data"].isel(channel=0).transpose("recording", "epoch", "sample")
        )
    out = xr.concat(traces, dim="event")
    return out.assign_coords(event=np.arange(len(traces)))


# Fit: ordinary least squares

def fit_weights(real_uv, traces, train_mask, fit_intercept=True):
    """Find the weights W[event, channel] that best explain the training EEG.

    We stack all valid (trial, sample) points of the training trials into rows:

        X : (n_points, n_events)    the bump values T_e(t) at those points
        Y : (n_points, n_channels)  the real EEG at those points (in microvolts)

    and solve  Y ~ X @ W  with `np.linalg.lstsq`, which returns the W that
    minimises the sum of squared errors  ||Y - X W||^2.
    All electrodes share the same X, so one call solves all of them at once.

    A point is used only if:
      - it belongs to a training trial,
      - the real EEG is not NaN on any electrode (valid sample before the RT),
      - the bumps are not NaN (same condition, since both come from epoch_data).

    The test trials are never touched here, so they can serve as an
    evaluation afterwards.

    Returns
    -------
    DataArray W with dims (event, channel).
    """
    # Put the axes in a known order: (recording, epoch, sample, channel/event)
    Y = real_uv["data"].transpose("recording", "epoch", "sample", "channel").values
    X = traces.transpose("recording", "epoch", "sample", "event").values
    n_ev = X.shape[-1]
    # True where the point is usable: no NaN anywhere AND in a training trial
    valid = (
        ~np.isnan(Y).any(axis=-1)
        & ~np.isnan(X).any(axis=-1)
        & train_mask.values[:, :, None]      # (recording, epoch) -> add sample axis
    )
    Xv, Yv = X[valid], Y[valid]  # now 2D
    if fit_intercept:
        Xv = np.column_stack([Xv, np.ones(len(Xv))])  # constant column

    coef, *_ = np.linalg.lstsq(Xv, Yv, rcond=None)  # (n_ev [+1], channel)

    W = xr.DataArray(coef[:n_ev], dims=("event", "channel"),
                     coords={"event": traces.event, "channel": real_uv.channel})
    b = (xr.DataArray(coef[n_ev], dims="channel",
                      coords={"channel": real_uv.channel}) if fit_intercept else None)
    return W, b

# Predict

def predict_least_square(traces, W, b=None):
    """Predicted EEG = sum over events of W[e, c] * T_e(t), for ALL trials.

    Returns a Dataset with a 'data' variable and dims
    (recording, epoch, channel, sample), i.e. the same layout as epoch_data,
    so it can be passed directly to `score(...)`.
    NaN stays NaN (unused trials, samples after the RT).
    """
    pred = xr.dot(traces, W, dim="event")      # (recording, epoch, sample, channel)
    if b is not None:
        pred = pred + b
    return xr.Dataset(
        {"data": pred.transpose("recording", "epoch", "channel", "sample")}
    )


# Helper to score only some trials
def only(ds, mask):
    """Keep only the trials where `mask` (recording, epoch) is True.

    All other trials are set to NaN. `score(...)` ignores NaN, so passing
    `only(ds, test_mask)` makes it score the test trials only.
    """
    out = ds.copy()
    out["data"] = ds["data"].where(mask)
    return out



def run_least_squares(epoch_data, weights, times, template_dict,
                      test_frac=0.2, seed=0, by="trial"):
    """Split, fit on the training trials and predict all trials.

    Returns
    -------
    dict with
      'W'         : fitted weights (event, channel), in microvolts
      'pred_uv'   : predicted EEG in microvolts (Dataset)
      'real_uv'   : real EEG in microvolts (Dataset)
      'test_mask' : True for the held-out trials (recording, epoch)

    Example
    -------
    out = run_least_squares(epoch_data, weights, times, pattern_assumption)
    test = out['test_mask']
    r2_test  = score(only(out['real_uv'], test),  only(out['pred_uv'], test),  "r2")
    r2_train = score(only(out['real_uv'], ~test), only(out['pred_uv'], ~test), "r2")
    """
    # Real EEG in microvolts
    real_uv = epoch_data.copy()
    real_uv["data"] = epoch_data["data"] * UV

    traces = event_traces(epoch_data, weights, times, template_dict)
    test_mask = split_trials(epoch_data, test_frac=test_frac, seed=seed, by=by)

    W, b = fit_weights(real_uv, traces, ~test_mask)
    pred_uv = predict_least_square(traces, W,b )

    return {"W": W,"b": b, "pred_uv": pred_uv, "real_uv": real_uv, "test_mask": test_mask}




# UPPER BOUND: separate weights for every trial (in-sample)

def fit_per_trial(real_uv, traces, min_points=None, fit_intercept=True):
    """Fit separate weights W[event, channel] for EACH trial, on that trial only.

    Same model as before, pred[c, t] = sum_e W[e, c] * T_e(t), but the
    least-squares problem is solved independently for every trial, using that
    trial's own EEG. The weights therefore see the data they are supposed to
    predict: this is NOT a prediction, it is an IN-SAMPLE UPPER BOUND on what
    the chosen templates (half-sine shape, width) and the event times can
    explain, if the topographies were perfectly tuned to each trial.

    Caveat: each trial has n_events * n_channels free weights (e.g. 4 * 32 = 128)
    fitted to one noisy trial, so part of the score is just noise being fitted.
    See below for what R2 score fitting pure noise would giveEven pure noise would
    give (see `noise_floor`).

    Parameters
    ----------
    real_uv : Dataset with the real EEG in microvolts (recording, epoch, channel, sample).
    traces  : DataArray from `event_traces` (event, recording, epoch, sample).
    min_points : minimum number of valid samples needed to fit a trial
                 (default: number of events (+1 with intercept); fewer points than
                 free parameters cannot be fitted).
    fit_intercept : if True (default), a constant is fitted per trial and electrode
                 in addition to the event weights. R2 is measured around the trial's
                 mean, so without an intercept any DC offset of the trial counts as
                 error and R2 can be strongly negative even in-sample. The intercept
                 is included in the returned prediction but not in W.

    Returns
    -------
    W       : DataArray (recording, epoch, event, channel), NaN for skipped trials.
    pred_ds : Dataset with 'data' (recording, epoch, channel, sample), same layout
              as epoch_data, NaN where the real data is NaN or the trial was skipped.
    """
    Y = real_uv["data"].transpose("recording", "epoch", "sample", "channel").values
    X = traces.transpose("recording", "epoch", "sample", "event").values
    n_rec, n_ep, n_s, n_ch = Y.shape
    n_ev = X.shape[-1]
    if min_points is None:
        min_points = n_ev + int(fit_intercept)   # n_ev + 1 if intercept = True

    W = np.full((n_rec, n_ep, n_ev, n_ch), np.nan)   # set w matrix dimension
    pred = np.full_like(Y, np.nan)                          # set pred matrix dimension

    for r in range(n_rec):       #for each subject
        for e in range(n_ep):     #for each event
            # valid samples of this trial: before the RT, no NaN anywhere
            valid = ~np.isnan(Y[r, e]).any(axis=-1) & ~np.isnan(X[r, e]).any(axis=-1)
            if valid.sum() < min_points:
                continue                     # unused trial, or too short to fit
            Xi, Yi = X[r, e][valid], Y[r, e][valid]
            if fit_intercept:
                Xi = np.column_stack([Xi, np.ones(len(Xi))])   # constant column
            Wi, *_ = np.linalg.lstsq(Xi, Yi, rcond=None)       # (event[+1], channel)
            W[r, e] = Wi[:n_ev]             #add output of least_square for subject r, event e to weight matrix
            pred[r, e][valid] = Xi @ Wi     #add prediction by computing Y = Xw to prediction matrix

    W = xr.DataArray(
        W, dims=("recording", "epoch", "event", "channel"),
        coords={"recording": real_uv.recording, "epoch": real_uv.epoch,
                "event": traces.event, "channel": real_uv.channel},
    )
    pred_ds = xr.Dataset({"data": xr.DataArray(
        pred, dims=("recording", "epoch", "sample", "channel"),
        coords={"recording": real_uv.recording, "epoch": real_uv.epoch,
                "sample": real_uv.sample, "channel": real_uv.channel},
    ).transpose("recording", "epoch", "channel", "sample")})
    return W, pred_ds


# R2 if only noise predicted

def noise_floor(real_uv, n_events):  # pass n_events only; the intercept costs one sample (n-1)
    """Rough R2 that fitting n_events regressors to pure noise would give.

    For a regression with k free regressors on n samples of noise, the expected
    in-sample R2 is about k / n. Returned per trial (recording, epoch), using the
    number of valid samples of each trial. If the per-trial upper bound is close
    to this value, the fit is mostly explained by overfitting noise.
    """
    n_valid = real_uv["data"].notnull().any("channel").sum("sample")
    return n_events / (n_valid - 1).where(n_valid > 1)   # -1: centering removes one dimension

def fit_weights_per_participant(real_uv, traces, train_mask, fit_intercept=True):
    """Same as fit_weights, but one least-squares fit PER PARTICIPANT
    (each participant gets its own scalp map per event)."""
    W_list, b_list = [], []
    for r in real_uv.recording.values:
        W_r, b_r = fit_weights(real_uv.sel(recording=[r]),
                               traces.sel(recording=[r]),
                               train_mask.sel(recording=[r]),
                               fit_intercept)
        W_list.append(W_r)
        b_list.append(b_r)
    W = xr.concat(W_list, dim="recording").assign_coords(recording=real_uv.recording.values)
    b = (xr.concat(b_list, dim="recording").assign_coords(recording=real_uv.recording.values)
         if fit_intercept else None)
    return W, b                                         # W: (recording, event, channel)

def run_least_squares_participants(epoch_data, weights, times, template_dict,
                      test_frac=0.2, seed=0, by="trial"):
    """Split, fit on the training trials and predict all trials.

    Returns
    -------
    dict with
      'W'         : fitted weights (event, channel), in microvolts
      'pred_uv'   : predicted EEG in microvolts (Dataset)
      'real_uv'   : real EEG in microvolts (Dataset)
      'test_mask' : True for the held-out trials (recording, epoch)

    Example
    -------
    out = run_least_squares(epoch_data, weights, times, pattern_assumption)
    test = out['test_mask']
    r2_test  = score(only(out['real_uv'], test),  only(out['pred_uv'], test),  "r2")
    r2_train = score(only(out['real_uv'], ~test), only(out['pred_uv'], ~test), "r2")
    """
    # Real EEG in microvolts
    real_uv = epoch_data.copy()
    real_uv["data"] = epoch_data["data"] * UV

    traces = event_traces(epoch_data, weights, times, template_dict)
    test_mask = split_trials(epoch_data, test_frac=test_frac, seed=seed, by=by)

    W, b = fit_weights_per_participant( real_uv, traces, ~test_mask)
    pred_uv = predict_least_square(traces, W,b )

    return {"W": W,"b": b, "pred_uv": pred_uv, "real_uv": real_uv, "test_mask": test_mask}
