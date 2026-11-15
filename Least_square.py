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

Here we instead learn ONE weight per (event, electrode), shared by all trials:

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

from helper_functions import simulate_from_hmpfit, split_trials

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
        w.values[0, :, e] = 1.0          # electrode 0, all trials, event e only
        sim = simulate_from_hmpfit(epoch_data, w, times, template_dict)
        traces.append(
            sim["data"].isel(channel=0).transpose("recording", "epoch", "sample")
        )
    out = xr.concat(traces, dim="event")
    return out.assign_coords(event=np.arange(len(traces)))


# Fit: ordinary least squares

def fit_weights(real_uv, traces, train_mask):
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

    The test trials are never touched here, so they can serve as an honest
    evaluation afterwards.

    Returns
    -------
    DataArray W with dims (event, channel).
    """
    # Put the axes in a known order: (recording, epoch, sample, channel/event)
    Y = real_uv["data"].transpose("recording", "epoch", "sample", "channel").values
    X = traces.transpose("recording", "epoch", "sample", "event").values

    # True where the point is usable: no NaN anywhere AND in a training trial
    valid = (
        ~np.isnan(Y).any(axis=-1)
        & ~np.isnan(X).any(axis=-1)
        & train_mask.values[:, :, None]      # (recording, epoch) -> add sample axis
    )

    W, *_ = np.linalg.lstsq(X[valid], Y[valid], rcond=None)   # (event, channel)

    return xr.DataArray(
        W,
        dims=("event", "channel"),
        coords={"event": traces.event, "channel": real_uv.channel},
    )



# Predict

def predict_least_square(traces, W):
    """Predicted EEG = sum over events of W[e, c] * T_e(t), for ALL trials.

    Returns a Dataset with a 'data' variable and dims
    (recording, epoch, channel, sample), i.e. the same layout as epoch_data,
    so it can be passed directly to `score(...)`.
    NaN stays NaN (unused trials, samples after the RT).
    """
    pred = xr.dot(traces, W, dim="event")      # (recording, epoch, sample, channel)
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

    W = fit_weights(real_uv, traces, ~test_mask)
    pred_uv = predict_least_square(traces, W)

    return {"W": W, "pred_uv": pred_uv, "real_uv": real_uv, "test_mask": test_mask}