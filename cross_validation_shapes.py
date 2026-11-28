"""cross_validation_shapes.py
===========================

New version of cross_validation.py, with DIFFERENT NAMES so both files can be
imported in the same notebook:

    old (cross_validation.py)   new (this file)
    -------------------------   ----------------------
    EventWidthCV                EventShapeCV
    kfold_masks                 shape_kfold_masks
    summarize_cv / best_widths  summarize_shape_cv / best_shapes
    plot_cv / make_templates    plot_shape_cv / EventShapeCV.templates_for

What is new: scalp maps averaged on the TRAINING trials (no leakage), and learned
shapes ("learned:25") next to half-sine widths.

Tune the shape (half-sine width, or a shape learned from the data) of each
event by cross-validation.

THE QUESTION
------------
"Which bump shape best describes each event?"  We let held-out REAL EEG decide:
the shape that predicts unseen trials best wins. Each event gets its own shape.

CANDIDATE SHAPES
----------------
    a number, e.g. 50    -> half-sine of that width in ms (HMP's assumption)
    "learned:25"         -> shape LEARNED from the training trials, +-25 samples
                            around the event (no regression: the EEG of the training
                            trials, aligned on the event and projected on the event's
                            scalp map, is averaged)

SCALP MAPS (weights)
--------------------
    maps="train_mean" (default): one map per participant and event, averaged over
                                 the TRAINING trials only -> no trial copies its own
                                 EEG, and validation/test trials are never used.
    maps="trial"               : old behaviour, one map per trial read from that
                                 trial's own EEG (leakage: scores are inflated).
                                 Kept only for comparison.
    maps="lstsq"               : maps (+ one offset per electrode) fitted by least
                                 squares on the TRAINING trials, per participant, all
                                 events jointly (overlapping events don't contaminate
                                 each other). Side analysis: these are no longer HMP's
                                 maps, but the best maps for the given event times and
                                 shapes.

THE PROCEDURE
-------------
1. A final TEST set of trials is set aside (`split_trials`) and never used for any choice.
2. On the remaining trials, K-fold cross-validation (`shape_kfold_masks`).
3. For event e and a candidate shape s (the other events keep their current shape),
   for every fold: maps (and learned shapes) are computed on the training trials,
   the validation trials are predicted, R2 is computed. Mean R2 over folds.
   The best shape of each event is the one with the highest mean R2.
4. `evaluate` computes maps (and learned shapes) on all non-test trials and scores
   the TEST set once.

THE MODEL
---------
pred[c, t] = sum_e W[e, c] * T_e(t)
    T_e(t)  : the bump of event e (half-sine or learned shape) centred on the HMP
              event time of that trial, clipped to [0, RT)
    W[e, c] : scalp map of event e (see SCALP MAPS)

USAGE (notebook)
----------------
    tuner = EventShapeCV(epoch_data, times, weights, epoch_data.sfreq)
    test_mask = split_trials(epoch_data, test_frac=0.2, seed=0)
    folds = shape_kfold_masks(epoch_data, k=5, seed=0, exclude=test_mask)
    grid = [20, 30, 50, 80, 120, 200, "learned:10", "learned:25", "learned:50"]
    cv, best = tuner.cross_validate(grid, folds, half_width="all", n_passes=3)
    plot_shape_cv(cv)
    res = tuner.evaluate(best, test_mask, half_width=12)
    templates = tuner.templates_for(best, ~test_mask)   # for simulate_from_hmpfit / plots

Everything is in microvolts.
"""
import numpy as np
import pandas as pd

UV = 1e6  # volts -> microvolts


# =============================================================================
# 1. Cross-validation folds
# =============================================================================
def _shape_mask_np(mask):
    """Turns a trial mask into a plain numpy True/False grid.
    DataArray (recording, epoch) or array -> boolean numpy array."""
    if hasattr(mask, "dims"):
        mask = mask.transpose("recording", "epoch").values
    return np.asarray(mask, dtype=bool)


def shape_kfold_masks(epoch_data, k=5, seed=0, by="trial", exclude=None):
    """Split the trials into K folds. Each trial gets a fold number (0 to K−1), and the function returns K pairs (train, validation)

    by      : "trial"       -> random trials (mixed across participants).
              "participant" -> whole participants per fold (needs >= k participants).
    exclude : boolean (recording, epoch), True = trial kept out (e.g. test trials).

    Returns a list of K tuples (train_mask, val_mask), boolean DataArrays (recording, epoch).
    """
    import xarray as xr

    # Step 1: which trials can be used (the "pool")
    # .notnull().any(["channel", "sample"]) → for each trial: at least one non-NaN value on any electrode at any time
    pool = (epoch_data["data"].notnull().any(["channel", "sample"])
            .transpose("recording", "epoch").values).copy()
    if exclude is not None:     # Removes the excluded trials
        pool &= ~_shape_mask_np(exclude)

    # Step 2: give each trial a fold number
    rng = np.random.default_rng(seed)       # random number generator, with a fixed seed so same split every time.
    fold = np.full(pool.shape, -1, dtype=int)             # -1 = not in the pool
    # a) by random trials
    if by == "trial":
        idx = np.argwhere(pool)
        if len(idx) < k:       # Fewer trials than folds → impossible, clear error.
            raise ValueError(f"only {len(idx)} trials available for {k} folds")
        for j, o in enumerate(rng.permutation(len(idx))):   # trials in random order.
            fold[tuple(idx[o])] = j % k     # remainder of the division by k
    # b) whole participants: Stricter (it tests generalisation to new person), but you need at least K participants
    elif by == "participant":
        recs = np.where(pool.any(axis=1))[0] # participants have at least one usable trial
        if len(recs) < k:
            raise ValueError(f"only {len(recs)} participants available for {k} folds")
        for j, r in enumerate(rng.permutation(recs)):
            fold[r, pool[r]] = j % k       # all the trials of one participant go into the same fold
    else: # Any other value for by → error
        raise ValueError("by must be 'trial' or 'participant'")

    # Step 3: build the (train, validation) pairs
    coords = {"recording": epoch_data.recording, "epoch": epoch_data.epoch}

    def wrap(m):
        return xr.DataArray(m, dims=("recording", "epoch"), coords=coords)

    # validation =  trials of fold f (fold == f)
    # train = all the other usable trials (fold >= 0 = in the pool, and fold != f = not in fold f)
    return [(wrap((fold >= 0) & (fold != f)), wrap(fold == f)) for f in range(k)]


# =============================================================================
# 2. Small helpers
# =============================================================================
def as_shapes(widths, n_events):
    """# turns whatever you give it into the same format: a dictionary {event: shape}
    Number, string, list or dict -> dict {event: shape}."""
    # a) already gave a dictionary → return a copy of it
    if isinstance(widths, dict):
        return dict(widths)
    # b) gave a single value (like 50, or a string like "learned:25") → the same shape for every event
    if np.isscalar(widths) or isinstance(widths, str):
        return {e: widths for e in range(n_events)}
    return dict(enumerate(widths))


def is_learned_shape(spec):
    return isinstance(spec, str) and spec.startswith("learned:")


def learned_half_width(spec):
    return int(spec.split(":")[1])


def shape_r2_per_channel(Y, P):
    """R2 per electrode, Y and P of shape (n_points, n_channels) (sklearn's definition)."""
    if len(Y) < 2:
        return np.full(Y.shape[1], np.nan)
    ss_res = ((Y - P) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 1 - ss_res / ss_tot


class ShapeTemplate:
    """Template learned from the data; usable in simulate_from_hmpfit (needs `.template`)."""
    def __init__(self, template):
        self.template = np.asarray(template, dtype=float)


# =============================================================================
# 3. The cross-validation engine
# =============================================================================
class EventShapeCV:
    """Holds the real EEG, the event times and the HMP weights in a compact form.

    Only samples from 0 to the RT of the trials HMP used are kept ("points"):
      Y[i]         : real EEG on all electrodes (microvolts)
      ri, ei, si   : participant, epoch (positions in epoch_data) and sample
    """

    def __init__(self, epoch_data, times, weights, sfreq, template_cls=None,
                 scale=UV, weights_scale=None, maps="train_mean"):
        """
        epoch_data : Dataset (recording, epoch, channel, sample); unused trials and
                     samples after the RT should be NaN.
        times      : hmp.utils.event_times(..., add_rt=True) (last column = RT).
        weights    : DataArray (channel, trial, event), PER TRIAL
                     (hmp.utils.event_channels(..., mean=False)).
        sfreq      : sampling frequency in Hz (use epoch_data.sfreq).
        maps       : "train_mean" (default) or "trial" (old behaviour, leakage).
        """
        if template_cls is None:
            from helper_functions import HalfSine as template_cls
        if maps not in ("train_mean", "trial", "lstsq"):
            raise ValueError("maps must be 'train_mean', 'trial' or 'lstsq'")
        self.sfreq, self.template_cls, self.maps = float(sfreq), template_cls, maps

        data = (epoch_data["data"].transpose("recording", "epoch", "channel", "sample")
                .values * scale)    # from Volt to microvolt
        R, E, C, S = data.shape
        self.shape = (R, E, S)

        # Event times and RT of every trial, on the (recording, epoch) grid
        other = [d for d in times.dims if d != "trial"][0]
        t = times.transpose("trial", other).values
        self.n_events = t.shape[1] - 1
        self.rec_idx = {str(r): i for i, r in enumerate(epoch_data.recording.values)}
        self.ep_idx = {int(e): i for i, e in enumerate(epoch_data.epoch.values)}
        self.ev_time = np.zeros((R, E, self.n_events), dtype=int)
        self.rt = np.zeros((R, E), dtype=int)              # 0 = trial not usable
        for i, (r, e) in enumerate(zip(times.recording.values, times.epoch.values)):
            ri, ei = self.rec_idx.get(str(r)), self.ep_idx.get(int(e))
            if ri is None or ei is None:
                continue
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

        # HMP weights per trial, on the (recording, epoch, event, channel) grid
        ws = scale if weights_scale is None else weights_scale
        self.W_trial = self._prepare_weights(weights, epoch_data, R, E, C) * ws

    def _prepare_weights(self, weights, epoch_data, R, E, C):
        """DataArray (channel, trial, event) -> array (R, E, n_events, C)."""
        # arranges the weights (HMP's scalp maps) in the same format as the EEG, so the map of any trial can be looked up easily.
        w = weights.reindex(channel=epoch_data.channel.values)
        if w.sizes["event"] != self.n_events:
            raise ValueError(f"{w.sizes['event']} events in weights, {self.n_events} in times")
        wv = w.transpose("trial", "event", "channel").values
        W = np.zeros((R, E, self.n_events, C))
        n_found = 0
        for i, (r, e) in enumerate(zip(weights.recording.values, weights.epoch.values)):
            ri, ei = self.rec_idx.get(str(r)), self.ep_idx.get(int(e))
            if ri is None or ei is None:
                continue
            W[ri, ei] = wv[i]
            n_found += 1
        if n_found != wv.shape[0]:
            print(f"warning: {wv.shape[0] - n_found} weight trials not in epoch_data")
        if np.isnan(W).any():
            raise ValueError("NaN in weights")
        return W

    # ---- scalp maps -----------------------------------------------------------
    def maps_from(self, train_mask):
        """Scalp maps (R, E, n_events, C) used to predict, given the training trials.

        maps="train_mean": per participant and event, the mean of the per-trial HMP
                           maps over the TRAINING trials, given to every trial.
        maps="trial"     : the per-trial maps (old behaviour, leakage).
        """
        if self.maps == "trial":
            return self.W_trial                    # ("lstsq" also starts from the train mean)
        train = _shape_mask_np(train_mask) & (self.rt > 0)
        W = np.zeros_like(self.W_trial)
        for r in range(self.shape[0]):
            if train[r].any():
                W[r] = self.W_trial[r][train[r]].mean(axis=0)   # (n_events, C), broadcast
        return W

    def fit_maps_lstsq(self, X, train_mask):
        """Least-squares maps on the training trials, per participant, all events jointly.

        Y ~ X @ B + b  ->  B (n_events, C) and b (C,) per participant.
        Returns W (R, E, n_events, C) and offsets (R, C).
        """
        train_pts = (_shape_mask_np(train_mask) & (self.rt > 0))[self.ri, self.ei]
        R, E, ne, C = self.W_trial.shape
        W = np.zeros((R, E, ne, C))
        b = np.zeros((R, C))
        for r in range(R):
            sel = train_pts & (self.ri == r)
            if sel.sum() <= ne + 1:
                continue
            Xr = np.column_stack([X[sel], np.ones(sel.sum())])        # + intercept
            coef = np.linalg.lstsq(Xr, self.Y[sel], rcond=None)[0]    # (ne + 1, C)
            W[r] = coef[:ne]
            b[r] = coef[ne]
        return W, b

    def _prediction_maps(self, X, W_mean, train_mask):
        """Maps (and offsets) used to predict: the train-mean maps, or least squares."""
        if self.maps == "lstsq":
            return self.fit_maps_lstsq(X, train_mask)
        return W_mean, None

    # ---- bump shapes ----------------------------------------------------------
    def _halfsine(self, width):
        key = float(width)
        if key not in self._tmpl:
            tmpl = np.asarray(self.template_cls(self.sfreq, width=width).template, dtype=float)
            # Guard against float noise: an exact number of samples gives exactly that many
            n = width * self.sfreq / 1000
            if abs(n - round(n)) < 1e-3:
                tmpl = tmpl[:int(round(n))]
            self._tmpl[key] = tmpl
        return self._tmpl[key]

    def learned_template(self, event, half_width, W, train_mask,
                         between_events=True, margin=4, min_frac=0.2):
        """Shape of `event` learned on the training trials (no regression).

        Every training point within +-half_width samples of the event is projected on
        the event's scalp map (one value per point), then averaged per lag.
        Normalised to 1 at the event time (like the half-sine peak).

        between_events : if True (default), in each trial only the samples BETWEEN the
                         previous and the next event are used (previous event = stimulus
                         onset for the first event, next event = RT for the last one),
                         minus `margin` samples on each side. This prevents the shape of
                         an event from absorbing its neighbours (e.g. event 0 picking up
                         events 1 and 2 with a +-400 ms window).
        margin         : samples kept away from the neighbouring events (default 4,
                         about half of a short bump at 256 Hz).
        min_frac       : lags covered by fewer than min_frac x the best-covered lag are
                         set to 0 (edges of the trial: too few trials, pure noise).
        """
        # Step 1: which points to use:
        train_pts = (_shape_mask_np(train_mask) & (self.rt > 0))[self.ri, self.ei]
        # True for the points that belong to a training trial used by HMP (rt > 0)

        t_ev = self.ev_time_pts[:, event] # for each point, time of this event in its own trial
        lag = self.si - t_ev #  distance between the point and the event
        sel = train_pts & (np.abs(lag) <= half_width) # Keeps training points within ±half_width samples of the event

        # Step 2: stay between the neighbouring events
        if between_events:
            # # time of previous event in the trial (for event 0: the stimulus, sample 0):
            prev_t = (self.ev_time_pts[:, event - 1] if event > 0
                      else np.zeros_like(t_ev))                         # stimulus onset
            # time of the next event (for the last event: the RT):
            next_t = (self.ev_time_pts[:, event + 1] if event < self.n_events - 1
                      else self.rt[self.ri, self.ei])                   # RT
            # allowed zone: from margin samples after the previous event, to margin samples before the next one
            lo = np.where(event > 0, prev_t + margin, prev_t)           # no margin at stimulus
            hi = np.where(event < self.n_events - 1, next_t - margin, next_t)
            sel &= (self.si >= np.minimum(lo, t_ev)) & (self.si <= np.maximum(hi, t_ev)) # Keeps only the points inside that zone

        # Step 3: one number per point, the projection on the map
        # For each kept point, the scalp map of this event (mean over the training trials):
        m = W[self.ri[sel], self.ei[sel], event]                        # (n_sel, C)

        # For each point: how much does the EEG on the 32 electrodes look like the event's map, and with which amplitude?
        with np.errstate(invalid="ignore", divide="ignore"):
            proj = np.einsum("pc,pc->p", self.Y[sel], m) / (m ** 2).sum(axis=1)
            # einsum("pc,pc->p", Y, m) = for each point, the sum over electrodes of EEG × map (a dot product);
            # divided by  sum of map² = normalised:
            # EEG exactly equal to the map gives 1, half of it gives 0.5, the opposite gives −1
        ok = np.isfinite(proj) # Removes the possible NaNs

        # Step 4: average by distance to the event
        n_lags = 2 * half_width + 1 # turns distances (−25…+25) into box numbers (0…50)
        # For each distance, sum of the projections of all points at that distance (all trials together):
        sums = np.bincount(lag[sel][ok] + half_width, weights=proj[ok], minlength=n_lags)
        # How many points fell at each distance:
        counts = np.bincount(lag[sel][ok] + half_width, minlength=n_lags)
        # Average projection at each distance:
        # = shape of the event over time (tc = "time course")
        # Distances with no points stay at 0.
        tc = np.divide(sums, counts, out=np.zeros(n_lags), where=counts > 0)

        # Step 5: clean up and normalise
        # Distances covered by too few points (less than 20 % of the best-covered distance) are set to 0:
        tc[counts < min_frac * counts.max()] = 0.0                      # too few trials
        # tc[half_width] = the value at the event (distance 0, in the middle):
        peak = tc[half_width] if tc[half_width] != 0 else 1.0
        # Divided by it, so shape = 1 at event (like  top of the half-sine)
        # Then shape × map gives exactly the map at the event
        return tc / peak

    def column(self, event, tmpl, key):
        """T_e(t) at every valid point for a given template (centred, clipped to [0, RT))."""
        if key in self._col:
            return self._col[key]
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

    def design(self, shapes, W, train_mask, fold_id):
        """Builds the matrix X, the input of _predict:
        One column per event, and in each column, the value of that event's bump at every point.
        Matrix X (n_points, n_events): one bump per event, with its own shape.
        Learned shapes are computed on `train_mask` (cached per fold)."""
        shapes = as_shapes(shapes, self.n_events)
        cols = []
        for e in range(self.n_events): #for each event
            s = shapes[e]
            # a) learned shape:
            if is_learned_shape(s):
                key = ("learned", e, learned_half_width(s), fold_id)
                # label for this column:
                # event e, window ±25, this fold (earned shape is computed from  training trials, and  differ in each fold)
                # -> used for the cache (self._col), so the column isn't recomputed every time.
                if key not in self._col:
                    # learns shape of event e on the training trials (train_mask), using the maps W
                    tmpl = self.learned_template(e, learned_half_width(s), W, train_mask)
                    # places this shape at the event's time in every trial, and stores result in the cache under key
                    self.column(e, tmpl, key)
                cols.append(self._col[key])
            # b) half-sine:
            else:
                cols.append(self.column(e, self._halfsine(s), ("hs", e, float(s))))
                # self._halfsine(s) = builds the half-sine of width s ms
                # self.column(...) = places it at the event's time in every trial
                # no fold_id: half-sine doesn't depend on  data, same in every fold and computed only once
        return np.column_stack(cols)

    # ---- windows, masks ---------------------------------------------------------
    def window(self, event, half_width):
        """Returns, for every point, True or False:
        True if the point is close to the given event, i.e. within ±half_width samples of it.
        Boolean over points: samples within +-half_width of the event time."""
        # self.si = the sample number of each point
        # self.ev_time_pts[:, event] = for each point, the time of this event in the point's trial
        return np.abs(self.si - self.ev_time_pts[:, event]) <= half_width

    def pts(self, trial_mask):
        """Converts a mask over trials into a mask over points.
        Boolean (recording, epoch) mask -> boolean mask over points."""
        return _shape_mask_np(trial_mask)[self.ri, self.ei]

    def _predict(self, X, sel, W, b=None):
        """Computes the predicted EEG at the selected points:
        for each point and each electrode, the sum over events of bump value × weight of that electrode.
        pred[c, t] = sum_e W[trial, e, c] * T_e(t) (+ offset b), on the points in `sel`."""
        Wp = W[self.ri[sel], self.ei[sel]]                 # (n_sel, n_events, C)
        # For each selected point, takes the scalp maps of its trial:
        # self.ri[sel] = its participant, self.ei[sel] = its trial

        P = np.einsum("pe,pec->pc", X[sel], Wp)
        # p = point, e = event, c = channel (electrode);
        # "pe,pec->pc" = for each point p and electrode c:
        # multiply bump value of event e by weight of event e on electrode c ] = X[p,e] x W[p,e,c]
        # then sum over e (the letter e disappears from the result)
        if b is not None: # With least squares only:
            P = P + b[self.ri[sel]]
            # adds each participant's offset per electrode ( intercept)
        return P

    # ---- cross-validation -------------------------------------------------------
    def cross_validate(self, widths_grid, folds, base_widths=50, half_width=None,
                       n_passes=1, verbose=True):
        """Find the best shape of each event (half-sine widths and/or learned shapes).

        widths_grid : candidates, e.g. [20, 50, 100, "learned:10", "learned:25"].
        folds       : output of `shape_kfold_masks`.
        base_widths : shapes of the other events at the start (number, string or dict).
        half_width  : half-size of the scoring window in samples, or "all" to score on
                      every sample before the RT. Default None: half of the widest
                      half-sine candidate.
        n_passes    : number of passes (events updated one after the other).

        Returns
        -------
        cv   : DataFrame, one row per (pass, event, shape, fold) and its R2.
        best : {event: best shape} from the last pass.
        """
        widths_grid = list(widths_grid)

        # defining scoring window around each event
        # half of the widest half-sine or 50/2, converted from ms to samples
        if half_width is None: # if no half_width given
            hs = [w for w in widths_grid if not is_learned_shape(w)] or [50]
            half_width = int(np.ceil(max(hs) / 1000 * self.sfreq / 2))
        current = as_shapes(base_widths, self.n_events)

        # each fold:
        # tr = training trials;
        # self.maps_from(tr) = scalp maps averaged over those training trials;
        # self.pts(va) = validation points
        fold_data = [(tr, self.maps_from(tr), self.pts(va)) for tr, va in folds]
        # before: val_pts = [self.pts(va) for _, va in folds]
        # train masks were not needed, as trials were thrown away (_): nothing was learned on train (no least square)

        if verbose:
            print(f"{len(folds)} folds, window = +-{half_width} samples, "
                  f"{len(widths_grid)} shapes, {self.n_events} events, maps = {self.maps}")

        # MAIN PART: search, event by event
        rows = [] # one row per (pass, event, shape, fold).
        previous = dict(current)    # keeps shapes from the previous pass, to know when to stop
        for p in range(1, n_passes + 1):    # Loop over the passes
            # One round over all the events: for each event in turn, try every candidate shape and keep the best
            for e in range(self.n_events):      # Loop over the events
                # decides which samples will count in the score ] score per trial or per event
                win = (np.ones(len(self.si), dtype=bool) if half_width == "all"
                       else self.window(e, half_width))
                for w in widths_grid:
                    cand = dict(current)
                    cand[e] = w     # shape of event e = w shape

                    # scoring a candidate:
                    for f, (tr, W_f, va) in enumerate(fold_data): # for each fold

                        X = self.design(cand, W_f, tr, f) # builds the bumps

                        W_p, b_p = self._prediction_maps(X, W_f, tr)
                        # maps used to predict: the train-mean maps (W_f), or the least-squares maps if maps="lstsq"
                        # b_p =  offsets (only for least squares, otherwise None)

                        sel = va & win # = validation points inside scoring window
                        # scoring:
                        r2 = np.nanmean(shape_r2_per_channel(self.Y[sel],
                                                             self._predict(X, sel, W_p, b_p)))
                        rows.append({"pass": p, "event": e, "shape": str(w),
                                     "fold": f, "r2": r2, "_spec": w})
                # keeping the best shape for this event:
                cv = pd.DataFrame(rows)
                m = (cv[(cv["pass"] == p) & (cv["event"] == e)]
                     .groupby("shape")["r2"].mean())
                # Table of results → only this pass and this event → mean R² over the folds, for each shape

                best_label = m.idxmax() # shape with the highest mean R² as text
                current[e] = next(w for w in widths_grid if str(w) == best_label) # original value in the grid (correct type)
                if verbose:
                    print(f"pass {p}, event {e}: best shape {current[e]} "
                          f"(mean R2 = {m.max():.4f})")
            if current == previous: # a) f no shape changed, stop
                break
            previous = dict(current) # b) remember new shapes and do another pass
        return pd.DataFrame(rows), dict(current)

    # ---- final evaluation -------------------------------------------------------
    def evaluate(self, widths, test_mask, half_width, train_mask=None):
        """Score the test trials. Maps and learned shapes are computed on `train_mask`
        (default: every usable trial that is not in the test set).

        Returns a dict with 'widths', 'r2_overall' (all test samples, mean over
        electrodes), 'r2_overall_per_channel', 'r2_event_windows' (event,).
        """
        # Defines the training trials for the final evaluation:
        # by default, every trial that isn't in the test set (the whole 80 %) ~ = "not"
        if train_mask is None:
            train_mask = ~_shape_mask_np(test_mask)      # turns a trial mask into a plain numpy True/False grid
                                                         # = Which trials to learn the shape from: only training trials
        widths = as_shapes(widths, self.n_events)
        W = self.maps_from(train_mask) # scalp maps = mean over the training trials, per participant
        X = self.design(widths, W, train_mask, "eval") # Builds the bumps
        W_p, b_p = self._prediction_maps(X, W, train_mask) # Predict: the mean maps (W), or the least-squares maps + offsets if maps="lstsq".

        # Test points → prediction → R² per electrode:
        te = self.pts(test_mask)
        overall = shape_r2_per_channel(self.Y[te], self._predict(X, te, W_p, b_p))

        # R² in a window around each event (test points only):
        per_event = []
        for e in range(self.n_events):
            sel = te & self.window(e, half_width)
            per_event.append(np.nanmean(shape_r2_per_channel(self.Y[sel],
                                                             self._predict(X, sel, W_p, b_p))))

        self._col = {k: v for k, v in self._col.items() if k[-1] != "eval"} # removing eval: not reusing learned shapes
        return {"widths": widths, "r2_overall": float(np.nanmean(overall)),
                "r2_overall_per_channel": overall,
                "r2_event_windows": np.array(per_event),
                "prediction": self._predict(X, te, W_p, b_p),     # (n_test_points, C)
                "test_points": te}

    def templates_for(self, widths, train_mask):
        """Gives the chosen shapes in a usable form.
        {event: template object} for simulate_from_hmpfit, learned shapes computed
        on `train_mask` (e.g. ~test_mask)."""
        widths = as_shapes(widths, self.n_events)
        W = self.maps_from(train_mask)
        out = {}
        for e, s in widths.items():
            if is_learned_shape(s):
                out[e] = ShapeTemplate(self.learned_template(e, learned_half_width(s), W, train_mask))
            else:
                out[e] = ShapeTemplate(self._halfsine(s))
        return out


# =============================================================================
# 4. Reading the results
# =============================================================================
def summarize_shape_cv(cv, pass_=None):
    """Mean, std and standard error of the CV R2 for every (event, shape)."""
    last = cv["pass"].max() if pass_ is None else pass_
    s = (cv[cv["pass"] == last].groupby(["event", "shape"])["r2"]
         .agg(["mean", "std", "count"]).reset_index())
    s["se"] = s["std"] / np.sqrt(s["count"])
    return s


def best_shapes(cv, pass_=None):
    """{event: shape with the highest mean CV R2}."""
    s = summarize_shape_cv(cv, pass_)
    out = {}
    for r in s.loc[s.groupby("event")["mean"].idxmax()].itertuples():
        out[int(r.event)] = r.shape
    return out


def _shape_order(labels):
    """Half-sine widths first (numeric order), then learned shapes."""
    num = sorted([l for l in labels if not l.startswith("learned")], key=float)
    lea = sorted([l for l in labels if l.startswith("learned")],
                 key=lambda l: int(l.split(":")[1]))
    return num + lea


def plot_shape_cv(cv, pass_=None):
    """One panel per event: mean CV R2 (+- standard error over folds) per shape,
    best shape in red."""
    import matplotlib.pyplot as plt
    s = summarize_shape_cv(cv, pass_)
    best = best_shapes(cv, pass_)
    events = sorted(s["event"].unique())
    fig, axes = plt.subplots(1, len(events), figsize=(3.6 * len(events), 3),
                             sharey=True, squeeze=False)
    for ax, e in zip(axes[0], events):
        d = s[s["event"] == e].set_index("shape").loc[_shape_order(s[s["event"] == e]["shape"].tolist())]
        x = np.arange(len(d))
        colors = ["red" if lab == best[e] else "C0" for lab in d.index]
        ax.errorbar(x, d["mean"], yerr=d["se"], fmt="none", ecolor="gray", capsize=3)
        ax.scatter(x, d["mean"], c=colors, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(d.index, rotation=90, fontsize=7)
        ax.set_title(f"Event {e}: best {best[e]}")
    axes[0][0].set_ylabel("CV R2")
    plt.tight_layout()
    return fig


def plot_shape_templates(templates, sfreq):
    """Plot the shape of each event (e.g. from tuner.templates_for(best, ~test_mask))."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    for e, t in templates.items():
        tm = np.asarray(t.template)
        lags = (np.arange(len(tm)) - len(tm) // 2) * 1000 / sfreq
        ax.plot(lags, tm, marker="o", ms=2, label=f"event {e}")
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("time from event (ms)")
    ax.set_ylabel("shape (1 at the event)")
    ax.legend()
    plt.tight_layout()
    return fig
