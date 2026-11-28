"""Tune the width of each event's half-sine by cross-validation.

THE QUESTION
------------
"How wide should the bump of each event be?"  We let held-out REAL EEG decide:
the width that predicts unseen trials best (R2 inside the event's window) wins.
Each event gets its own width.

THE PROCEDURE
-------------
1. A final TEST set of trials is set aside (use `split_trials`) and never used
   for any choice.
2. On the remaining trials, K-fold cross-validation (`kfold_masks`).
3. For event e and a candidate width w (all other events stay at a baseline width):
      - build the bumps T_e(t) with that width,
      - use fixed HMP weights,
      - predict the held-out fold and compute R2 on the samples inside the
        window around event e (same window for every candidate).
   The R2 is averaged over the folds. The best width of each event is the one
   with the highest mean R2 (`best_widths`).
4. `evaluate` refits on all non-test trials with the chosen widths and scores the
   TEST set, so the tuned widths can be compared with the baseline widths.

THE MODEL USED INSIDE THE LOOP
------------------------------
pred[c, t] = sum_e W[e, c] * T_e(t)

USAGE (notebook)
----------------
    tuner = EventWidthCV(epoch_data, times, SFREQ)
    test_mask = split_trials(epoch_data, test_frac=0.2, seed=0)
    folds = kfold_masks(epoch_data, k=5, seed=0, exclude=test_mask)
    cv, best = tuner.cross_validate([20, 30, 40, 50, 60, 80, 100], folds, n_passes=2)
    plot_cv(cv)
    templates = make_templates(best, SFREQ)       # for simulate_from_hmpfit
    tuner.evaluate(best, ~test_mask, test_mask, half_width=6)

Everything is in microvolts (the real EEG is multiplied by `scale`).
`template_cls` can be changed later to try another pattern shape: it only needs
to accept (sfreq, width=...) and expose `.template` (a 1D array).
"""
import numpy as np
import pandas as pd

UV = 1e6  # volts -> microvolts


# =============================================================================
# 1. Cross-validation folds
# =============================================================================
def _mask_np(mask):
    """DataArray (recording, epoch) or array -> boolean numpy array."""
    if hasattr(mask, "dims"):
        mask = mask.transpose("recording", "epoch").values
    return np.asarray(mask, dtype=bool)


def kfold_masks(epoch_data, k=5, seed=0, by="trial", exclude=None):
    """Split the trials into K folds.

    Parameters
    ----------
    epoch_data : Dataset (recording, epoch, channel, sample); trials without data
                 (rejected, NaN) are never used.
    k          : number of folds.
    seed       : random seed, so the folds are reproducible.
    by         : "trial"       -> random trials (mixed across participants).
                 "participant" -> whole participants per fold (stricter; needs
                                  at least k participants in the pool).
    exclude    : boolean (recording, epoch), True = trial kept out of the folds
                 (e.g. the final test trials).

    Returns
    -------
    List of K tuples (train_mask, val_mask), boolean DataArrays (recording, epoch).
    Fold i: val = the trials of fold i, train = all the other pool trials.
    """
    import xarray as xr
    has = (epoch_data["data"].notnull().any(["channel", "sample"])
           .transpose("recording", "epoch").values)
    pool = has.copy()
    if exclude is not None:
        pool &= ~_mask_np(exclude)

    rng = np.random.default_rng(seed)
    fold = np.full(pool.shape, -1, dtype=int)             # -1 = not in the pool
    if by == "trial":
        idx = np.argwhere(pool)
        if len(idx) < k:
            raise ValueError(f"only {len(idx)} trials available for {k} folds")
        for j, o in enumerate(rng.permutation(len(idx))):
            fold[tuple(idx[o])] = j % k
    elif by == "participant":
        recs = np.where(pool.any(axis=1))[0]
        if len(recs) < k:
            raise ValueError(f"only {len(recs)} participants available for {k} folds")
        for j, r in enumerate(rng.permutation(recs)):
            fold[r, pool[r]] = j % k
    else:
        raise ValueError("by must be 'trial' or 'participant'")

    coords = {"recording": epoch_data.recording, "epoch": epoch_data.epoch}

    def wrap(m):
        return xr.DataArray(m, dims=("recording", "epoch"), coords=coords)

    return [(wrap((fold >= 0) & (fold != f)), wrap(fold == f)) for f in range(k)]


# =============================================================================
# 2. Small helpers
# =============================================================================
def as_widths(widths, n_events):
    """Number, list or dict -> dict {event: width in ms}."""
    if isinstance(widths, dict):
        return dict(widths)
    if np.isscalar(widths):
        return {e: widths for e in range(n_events)}
    return dict(enumerate(widths))


def r2_per_channel(Y, P):
    """R2 per electrode, Y and P of shape (n_points, n_channels).

    Same definition as sklearn's r2_score: 1 - SS_res / SS_tot, with SS_tot
    centred on the mean of Y over the points given. NaN if fewer than 2 points.
    """
    if len(Y) < 2:
        return np.full(Y.shape[1], np.nan)
    ss_res = ((Y - P) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - ss_res / ss_tot


def make_templates(widths, sfreq, template_cls=None):
    """{event: width in ms} -> {event: template object}, the format expected by
    `simulate_from_hmpfit` (e.g. widths=best from `cross_validate`)."""
    if template_cls is None:
        from helper_functions import HalfSine as template_cls
    return {e: template_cls(sfreq, width=w) for e, w in widths.items()}


# =============================================================================
# 3. The cross-validation engine
# =============================================================================
class EventWidthCV:
    """Holds the real EEG and the event times in a compact form, and runs the
    cross-validation of the bump width of each event.

    Only the samples that matter are kept: samples from 0 to the RT of the trials
    that HMP used and that have data. Every such sample is a "point" with
      Y[i]            : the real EEG on all electrodes (microvolts)
      ri[i], ei[i]    : its participant and epoch (position in epoch_data)
      si[i]           : its sample number
    so train/validation/test masks (recording, epoch) are turned into masks over
    points with a simple lookup.
    """

    def __init__(self, epoch_data, times, weights, sfreq, template_cls=None,
                 scale=UV, weights_scale=None):
        """
        epoch_data : Dataset (recording, epoch, channel, sample); unused trials
                     and samples after the RT should be NaN (as in the loading code).
        times      : DataArray from hmp.utils.event_times(..., add_rt=True); one row
                     per trial, one column per event, last column = RT, with
                     `recording` and `epoch` coordinates on the trial axis.
        sfreq      : sampling frequency in Hz.
        weights : DataArray (channel, trial, event), with `recording` and `epoch`
              coordinates on the trial axis. Reordered to match
              epoch_data.channel.
        """
        if template_cls is None:
            from helper_functions import HalfSine as template_cls
        self.sfreq, self.template_cls = sfreq, template_cls

        data = (epoch_data["data"].transpose("recording", "epoch", "channel", "sample")
                .values * scale)
        R, E, C, S = data.shape
        self.shape = (R, E, S)

        # Event times and RT of every trial, placed on the (recording, epoch) grid
        other = [d for d in times.dims if d != "trial"][0]
        t = times.transpose("trial", other).values
        self.n_events = t.shape[1] - 1                     # last column = RT
        self.rec_idx = {str(r): i for i, r in enumerate(epoch_data.recording.values)}
        self.ep_idx = {int(e): i for i, e in enumerate(epoch_data.epoch.values)}
        self.ev_time = np.zeros((R, E, self.n_events), dtype=int)
        self.rt = np.zeros((R, E), dtype=int)              # 0 = trial not usable
        for i, (r, e) in enumerate(zip(times.recording.values, times.epoch.values)):
            ri, ei = self.rec_idx.get(str(r)), self.ep_idx.get(int(e))
            if ri is None or ei is None:
                continue                                   # trial of another participant
            self.ev_time[ri, ei] = t[i, :-1].astype(int)
            self.rt[ri, ei] = int(t[i, -1])
        self._trials = np.argwhere(self.rt > 0)

        # Valid points: before the RT, no NaN on any electrode
        before_rt = np.arange(S)[None, None, :] < self.rt[:, :, None]
        valid = ~np.isnan(data).any(axis=2) & before_rt
        self.ri, self.ei, self.si = np.nonzero(valid)
        self.Y = data[self.ri, self.ei, :, self.si]        # (n_points, n_channels)
        self.ev_time_pts = self.ev_time[self.ri, self.ei]  # (n_points, n_events)
        self.channels = epoch_data.channel.values
        self._tmpl, self._col = {}, {}

        # ---- fixed HMP weights, on the (recording, epoch, event, channel) grid
        ws = scale if weights_scale is None else weights_scale
        self.W = self._prepare_weights(weights, epoch_data, R, E, C) * ws


    def _prepare_weights(self, weights, epoch_data, R, E, C):
        """DataArray (channel, trial, event) -> array (R, E, n_events, C)."""
        w = weights.reindex(channel=epoch_data.channel.values)  # same electrode order
        if w.sizes["event"] != self.n_events:
            raise ValueError(f"{w.sizes['event']} events in weights, "
                             f"{self.n_events} in times")
        w = w.transpose("trial", "event", "channel").values
        W = np.zeros((R, E, self.n_events, C))
        n_found = 0
        for i, (r, e) in enumerate(zip(weights.recording.values, weights.epoch.values)):
            ri, ei = self.rec_idx.get(str(r)), self.ep_idx.get(int(e))
            if ri is None or ei is None:
                continue
            W[ri, ei] = w[i]
            n_found += 1
        if n_found != w.shape[0]:
            print(f"warning: {w.shape[0] - n_found} weight trials not in epoch_data")
        if np.isnan(W).any():
            raise ValueError("NaN in weights")
        return W

    # ---- bumps -------------------------------------------------------------
    def _template(self, width):
        key = float(width)
        if key not in self._tmpl:
            self._tmpl[key] = np.asarray(
                self.template_cls(self.sfreq, width=width).template, dtype=float)
        return self._tmpl[key]

    def column(self, event, width):
        """T_e(t) of one event for a bump of `width` ms, at every valid point.

        The template (length L, peak in the middle) is centred on the event time
        and clipped to [0, RT): the part that falls before sample 0 or after the
        RT is cut off. Same placement as `simulate_from_hmpfit`.
        """
        key = (event, float(width))
        if key in self._col:
            return self._col[key]
        tmpl = self._template(width)
        L = len(tmpl)
        trace = np.zeros(self.shape)
        for ri, ei in self._trials:
            c, rt = int(self.ev_time[ri, ei, event]), int(self.rt[ri, ei])
            onset = c - L // 2
            offset = c + L // 2 + 1
            if L % 2 == 0:
                offset -= 1                               # keep exactly L samples
            on, off = max(onset, 0), min(offset, rt, self.shape[2])
            if off > on:
                trace[ri, ei, on:off] = tmpl[on - onset: L - (offset - off)]
        col = trace[self.ri, self.ei, self.si]
        self._col[key] = col
        return col

    def design(self, widths):
        """Matrix X (n_points, n_events): one bump per event, with its own width."""
        widths = as_widths(widths, self.n_events)
        return np.column_stack([self.column(e, widths[e]) for e in range(self.n_events)])

    # ---- windows, masks ----------------------------------------------------
    def window(self, event, half_width):
        """Boolean over points: samples within +-half_width of the event time."""
        return np.abs(self.si - self.ev_time_pts[:, event]) <= half_width

    def pts(self, trial_mask):
        """Boolean (recording, epoch) mask -> boolean mask over points."""
        return _mask_np(trial_mask)[self.ri, self.ei]


    def _predict(self, X, sel):
        """pred[c, t] = sum_e W[trial, e, c] * T_e(t), on the points in `sel`.
        Weights are looked up per trial, only for the selected points."""
        Wp = self.W[self.ri[sel], self.ei[sel]]            # (n_sel, n_events, C)
        return np.einsum("pe,pec->pc", X[sel], Wp)

    # ---- cross-validation --------------------------------------------------
    def cross_validate(self, widths_grid, folds, base_widths=50, half_width=None,
                       n_passes=1, verbose=True):
        """Find the best width of each event.

                For every event e and candidate width w: the other events keep their
                current width, the weights are the fixed HMP weights; the folds only give a spread of R² over subsets of trials,
                and R2 (mean over electrodes) is computed on the validation trials, only
                on the samples inside the window of event e.

                Parameters
                ----------
                widths_grid : candidate widths in ms (same list for every event).
                folds       : output of `kfold_masks`.
                base_widths : widths of the other events at the start (number or dict).
                half_width  : half-size of the scoring window in samples, or "all" to score on
                              every sample before the RT (one shared score for all events, so
                              the search converges). Default None: half of the widest candidate.
                n_passes    : number of passes. In pass 1 every event is searched with the
                              others at `base_widths`. Events are updated one after the other,
                              so each event already sees the new widths of the previous ones;
                              in pass 2 and later, all other events sit at their latest best
                              widths.

                Returns
                -------
                cv   : DataFrame with one row per (pass, event, width, fold) and its R2.
                best : {event: best width} from the last pass.
                """
        widths_grid = list(widths_grid)

        if half_width is None:
            half_width = int(np.ceil(max(widths_grid) / 1000 * self.sfreq / 2))
        current = as_widths(base_widths, self.n_events)
        val_pts = [self.pts(va) for _, va in folds]  # train masks not needed
        if verbose:
            print(f"{len(folds)} folds, window = +-{half_width} samples, "
                  f"{len(widths_grid)} widths, {self.n_events} events")

        rows = []
        previous = dict(current)
        for p in range(1, n_passes + 1):
            for e in range(self.n_events):
                win = (np.ones(len(self.si), dtype=bool) if half_width == "all"
                       else self.window(e, half_width))
                for w in widths_grid:
                    cand = dict(current)
                    cand[e] = w
                    X = self.design(cand)
                    P = self._predict(X, win)  # predictions in the window
                    Yw = self.Y[win]
                    for f, va in enumerate(val_pts):
                        v = va[win]
                        r2 = np.nanmean(r2_per_channel(Yw[v], P[v]))
                        rows.append({"pass": p, "event": e, "width_ms": w,
                                     "fold": f, "r2": r2})
                cv = pd.DataFrame(rows)
                m = (cv[(cv["pass"] == p) & (cv["event"] == e)]
                     .groupby("width_ms")["r2"].mean())
                current[e] = m.idxmax()
                if verbose:
                    print(f"pass {p}, event {e}: best width {current[e]} ms "
                      f"(mean R2 = {m.max():.4f})")
            if current == previous:
                break
            previous = dict(current)
        cv = pd.DataFrame(rows)
        return cv, dict(current)

    # ---- final evaluation --------------------------------------------------
    def evaluate(self, widths, test_mask, half_width):
        """Score the test trials with these widths and the HMP weights.

        Returns a dict with
          'r2_overall'        : R2 on all test samples (pooled), mean over electrodes
          'r2_event_windows'  : array (event,), R2 inside each event's window
          'r2_overall_per_channel', 'coef', 'widths'
        """
        widths = as_widths(widths, self.n_events)
        X = self.design(widths)
        te = self.pts(test_mask)
        overall = r2_per_channel(self.Y[te], self._predict(X, te))
        per_event = []
        for e in range(self.n_events):
            sel = te & self.window(e, half_width)
            per_event.append(np.nanmean(r2_per_channel(self.Y[sel],
                                                       self._predict(X, sel))))
        return {"widths": widths, "r2_overall": float(np.nanmean(overall)),
                "r2_overall_per_channel": overall,
                "r2_event_windows": np.array(per_event)}


# =============================================================================
# 4. Reading the results
# =============================================================================
def summarize_cv(cv, pass_=None):
    """Mean, std and standard error of the CV R2 for every (event, width)."""
    last = cv["pass"].max() if pass_ is None else pass_
    s = (cv[cv["pass"] == last].groupby(["event", "width_ms"])["r2"]
         .agg(["mean", "std", "count"]).reset_index())
    s["se"] = s["std"] / np.sqrt(s["count"])
    return s


def best_widths(cv, pass_=None):
    """{event: width with the highest mean CV R2}."""
    s = summarize_cv(cv, pass_)
    out = {}
    for r in s.loc[s.groupby("event")["mean"].idxmax()].itertuples():
        w = float(r.width_ms)
        out[int(r.event)] = int(w) if w.is_integer() else w
    return out


def plot_cv(cv, pass_=None):
    """One panel per event: mean CV R2 (+- standard error over folds) against the
    width, best width in red."""
    import matplotlib.pyplot as plt
    s = summarize_cv(cv, pass_)
    best = best_widths(cv, pass_)
    events = sorted(s["event"].unique())
    fig, axes = plt.subplots(1, len(events), figsize=(3.2 * len(events), 2.8),
                             sharey=True, squeeze=False)
    for ax, e in zip(axes[0], events):
        d = s[s["event"] == e]
        ax.errorbar(d["width_ms"], d["mean"], yerr=d["se"], marker="o", capsize=3)
        ax.axvline(best[e], color="red", ls="--", lw=1)
        ax.set_title(f"Event {e}: best {best[e]} ms")
        axes[0][0].set_ylabel("CV R2 (all samples)")
    axes[0][0].set_ylabel("CV R2 (event window)")
    plt.tight_layout()
    return fig