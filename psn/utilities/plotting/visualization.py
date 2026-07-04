"""PSN visualization - matches MATLAB visualization.m exactly"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec


def _subsample_for_imshow(M, max_side=2048):
    """Strided-subsample a (possibly huge) 2D array for display.

    For an (N, N) matrix at N=25000, imshow holds a 5 GB array and resamples it
    internally (slow), and a 99th-percentile clim over it sorts ~N^2 elements.
    Subsampling to <= max_side per axis FIRST makes both cheap. Strided slicing
    is a free view (no full read) and keeps the ACTUAL values, so a clim
    computed from the result matches the original. Returns (M_sub, extent) where
    extent maps the drawn image back onto the original 0..N axes - pass it to
    imshow(..., extent=extent) so the axis labels are unchanged.
    """
    M = np.asarray(M)
    h, w = M.shape[:2]
    sh = max(1, int(np.ceil(h / max_side)))
    sw = max(1, int(np.ceil(w / max_side)))
    M_sub = M[::sh, ::sw] if (sh > 1 or sw > 1) else M
    return M_sub, [0, w, h, 0]


def cmapsign4(n=256):
    """
    Return a cyan-blue-black-red-yellow colormap.

    This colormap is symmetric around black (center), going from
    cyan-white through cyan and blue to black, then from black
    through red and yellow to yellow-white.

    This is useful for visualizing data that has both positive and
    negative values, with zero mapped to black.

    Parameters
    ----------
    n : int, optional
        Number of colors in the colormap. Default: 256.

    Returns
    -------
    cmap : LinearSegmentedColormap
        A matplotlib colormap object.
    """
    colors = [
        (0.8, 1, 1),    # cyan-white
        (0, 1, 1),      # cyan
        (0, 0, 1),      # blue
        (0, 0, 0),      # black (center)
        (1, 0, 0),      # red
        (1, 1, 0),      # yellow
        (1, 1, 0.8),    # yellow-white
    ]
    cmap = LinearSegmentedColormap.from_list('cmapsign4', colors, N=n)
    return cmap


def redblue(n=256):
    """
    Return a red-blue diverging colormap.

    This colormap transitions from blue through white to red.
    Useful for visualizing symmetric data centered at zero
    (e.g., correlation matrices, covariance matrices, residuals).

    Parameters
    ----------
    n : int, optional
        Number of colors in the colormap. Default: 256.

    Returns
    -------
    cmap : LinearSegmentedColormap
        A matplotlib colormap object.
    """
    mid = n // 2

    # Build color array
    colors_list = []

    # Blue to white (first half)
    for i in range(mid):
        t = i / (mid - 1) if mid > 1 else 1
        colors_list.append((t, t, 1.0))  # R, G increase; B stays 1

    # White to red (second half)
    for i in range(n - mid):
        t = i / (n - mid - 1) if (n - mid) > 1 else 1
        colors_list.append((1.0, 1.0 - t, 1.0 - t))  # R stays 1; G, B decrease

    cmap = LinearSegmentedColormap.from_list('redblue', colors_list, N=n)
    return cmap


def _draw_axis_break(ax, x0):
    """Draw a little '//' break mark on the bottom spine at data-x ``x0``, to
    signal that the x-axis scale changes there (e.g. ordinary log -> mirror-log
    at the head/tail junction, or linear -> inverse-log). Convention: whenever
    the flow of the x ticks is rescaled, mark the transition."""
    try:
        xa = ax.transAxes.inverted().transform(ax.transData.transform((x0, 0)))[0]
    except Exception:
        return
    if not (0.0 < xa < 1.0):
        return
    d = 0.014
    kw = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1.0, zorder=10)
    for off in (-0.6 * d, 0.6 * d):
        ax.plot((xa - d + off, xa + d + off), (-d, d), **kw)


def _set_headtail_log_xscale(ax, ndims, tail=10, zero_at=None, rotation=0):
    """Apply a 'head-tail log' x-scale to a dimension axis.

    Ordinary log spacing for dimensions 1..ndims-tail, then the final ``tail``
    dimensions are spaced as a MIRROR IMAGE (inverse log) of the first ``tail``
    - so the leading AND the trailing dimensions stay legible. On a plain log
    axis the last dimensions are crushed against the right edge; here they fan
    back out, mirroring the head.

    Also sets integer ticks (head nice-values + junction + a few mirrored tail
    dims) and the x-limits. ``zero_at`` (e.g. 0.5) is the placeholder position
    where a K=0 point is drawn; when given it gets a '0' tick. Falls back to a
    plain log scale when there aren't enough dimensions to carve out a tail.
    """
    ndims = int(round(float(ndims)))
    left = (zero_at * 0.8) if zero_at else 0.8
    if ndims <= 2 * tail + 1:
        ax.set_xscale('log')
        ax.set_xlim([left, ndims + 0.5])
        if rotation:
            ax.tick_params(axis='x', rotation=rotation)
        return

    N = float(ndims)
    j = N - tail                       # head/tail junction dimension
    logj = np.log10(j)
    logt = np.log10(tail + 1.0)

    def fwd(x):
        x = np.asarray(x, dtype=float)
        head = np.log10(np.clip(x, 1e-9, None))
        # tail: log10(j) + log10(tail+1) - log10(N - x + 1), mirroring the head
        tailv = logj + logt - np.log10(np.clip(N - x + 1.0, 1e-3, None))
        return np.where(x <= j, head, tailv)

    def inv(y):
        y = np.asarray(y, dtype=float)
        head = 10.0 ** y
        tailx = N + 1.0 - j * (tail + 1.0) / (10.0 ** y)
        return np.where(y <= logj, head, tailx)

    ax.set_xscale('function', functions=(fwd, inv))
    ax.set_xlim([left, N + 0.5])

    tick_vals, tick_labels = [], []
    if zero_at:
        tick_vals.append(zero_at)
        tick_labels.append('0')
    # Head: nice values, but leave a margin before the junction so their labels
    # don't collide with it (they bunch up in transformed space near j).
    for tk in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000,
               10000, 20000, 50000):
        if 1 <= tk < j and (logj - np.log10(tk)) > 0.18:
            tick_vals.append(tk)
            tick_labels.append(str(tk))
    jj = int(round(j))
    tick_vals.append(jj)
    tick_labels.append(str(jj))
    # Tail: just the endpoint N (multi-digit labels crowd the narrow tail, so
    # only the junction + N are shown).
    if N - 0 > jj:
        tick_vals.append(int(round(N)))
        tick_labels.append(str(int(round(N))))
    seen, tv, tl = set(), [], []
    for v, lab in zip(tick_vals, tick_labels):
        if v not in seen:
            seen.add(v)
            tv.append(v)
            tl.append(lab)
    ax.set_xticks(tv)
    ax.set_xticklabels(tl, fontsize=6, rotation=rotation,
                       ha='right' if rotation else 'center')
    ax.minorticks_off()
    _draw_axis_break(ax, j)        # mark the log -> mirror-log junction


def _set_tail_invlog_xscale(ax, split=0.9, expand=0.4, K=99):
    """Inverse-log x-axis for a fraction in [0, 1].

    Linear on [0, split]; the [split, 1] tail is expanded (inverse-log, mirror
    of an ordinary log) so the data crowding near 1.0 - where the tradeoff
    curves and the chosen / Wiener / trial-avg points pile up - fans out and
    becomes legible. Same spirit as the head-tail dimension axis.

    split: where the linear region ends and the inverse-log tail begins.
    expand: transformed-space width given to the [split, 1] tail (vs its 1-split
            linear width) - larger = more magnification of the approach to 1.
    """
    a = float(split)
    W = float(expand)
    lk = np.log10(1.0 + K)
    aW = a + W
    # Slope of the (very steep) inverse-log right at x=1, used to extend the
    # transform linearly for x just past 1 - so the right xlim can sit a small
    # margin beyond x=1 and the markers there (trial-avg / chosen) aren't clipped.
    slope = W * K / ((1.0 - a) * np.log(10.0) * lk)

    def fwd(x):
        x = np.asarray(x, dtype=float)
        d = np.clip((1.0 - x) / (1.0 - a), 0.0, 1.0)     # 1 at split, 0 at x=1
        tail = a + W * (1.0 - np.log10(1.0 + K * d) / lk)
        over = aW + slope * (x - 1.0)                    # linear past x=1
        return np.where(x <= a, x, np.where(x <= 1.0, tail, over))

    def inv(T):
        T = np.asarray(T, dtype=float)
        val = np.power(1.0 + K, 1.0 - (T - a) / W)
        tail = 1.0 - (val - 1.0) / K * (1.0 - a)
        over = 1.0 + (T - aW) / slope
        return np.where(T <= a, T, np.where(T <= aW, tail, over))

    ax.set_xscale('function', functions=(fwd, inv))
    # Right edge a small transformed-space margin past x=1 so the x=1 markers
    # have clearance from the box.
    right = float(inv(np.array([aW + 0.06]))[0])
    ax.set_xlim(-0.01, right)
    ticks = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{t:g}' for t in ticks], fontsize=7,
                       rotation=45, ha='right')
    ax.minorticks_off()
    _draw_axis_break(ax, a)        # mark the linear -> inverse-log split


def _legend(ax, *args, **kwargs):
    """ax.legend(...) but placed BEHIND the plotted data (low zorder) so it never
    obscures curves/markers where they overlap it. This is the default for every
    legend in the diagnostic figure."""
    leg = ax.legend(*args, **kwargs)
    if leg is not None:
        leg.set_zorder(0.5)        # below data lines (2) and markers (>=3)
    return leg


def _plot_recovery_tradeoff(ax, rec):
    """Draw the recovery / bias-variance tradeoff panel from PRECOMPUTED data
    (results['recovery_tradeoff'], built in psn() by compute_recovery_tradeoff).
    Pure rendering - no heavy compute here; whichever curves/points the policy
    produced are drawn, the rest are simply absent."""
    ax.set_title('Recovery vs. signal retained\n(solid = analytic recovery, dashed = split-half r)')
    if not rec:
        ax.text(0.5, 0.5, 'no recovery data', ha='center', va='center',
                transform=ax.transAxes)
        return
    # Two y-axes: LEFT = analytic recovery (the quantity used to pick the threshold,
    # solid); RIGHT = empirical split-half reliability (validation, dashed).
    ax_r = ax.twinx()
    yL, yR = [], []                    # left (analytic) / right (split-half) y-values
    # Color convention: analytic-recovery color (left axis) vs split-half color (right
    # axis); markers reuse them. Shape: star = PSN chosen, square = trial-average.
    analytic_endpoint = analytic_endpoint_x = None   # analytic recovery at full retention
    prim_asf = prim_ar = None                        # primary basis analytic curve (for the chord)

    # Faint per-unit split-half traces + markers -> right axis.
    ut = rec.get('unit_traces')
    if ut is not None:
        xs_u = np.asarray(ut['sv_frac'])
        u_curves = np.asarray(ut['split_half_r'])          # (n_sub, nKs)
        yR.append(u_curves.ravel())
        h_units = None
        for row in u_curves:
            line, = ax_r.plot(xs_u, row, lw=0.5, color=[0.5, 0.5, 0.5], alpha=0.3, zorder=1)
            if h_units is None:
                h_units = line
        if h_units is not None:
            h_units.set_label('units')
        mk = ut.get('markers') or {}
        mx = np.asarray(mk.get('sv_frac', []))
        my = np.asarray(mk.get('split_half_r', []))
        if my.size:
            yR.append(my)
        grp = mk.get('group')
        if mx.size:
            if grp is not None:
                groups = np.asarray(grp)
                uniq = np.unique(groups)
                gcolors = plt.cm.hsv(np.linspace(0, 0.9, len(uniq)))
                g2c = {g: gcolors[i] for i, g in enumerate(uniq)}
                cols = [g2c[g] for g in groups]
                ax_r.scatter(mx, my, s=18, c=cols, alpha=0.65, zorder=4)
            else:
                ax_r.scatter(mx, my, s=18, color=[1, 0.3, 0.3], alpha=0.65, zorder=4)

    # Per-basis colors as (analytic solid, split-half dashed). The analytic (solid)
    # keeps the basis identity color (blue = signal, green = difference) so it
    # matches the fig6 objective trace. The signal split-half (dashed) is GOLD to
    # match the gold "TAvg vs Denoised" dots in the split-half reliability panel
    # (fig14); the difference split-half is purple so the two dashed curves stay
    # distinguishable in compare mode.
    GOLD = (1.0, 0.84, 0.0)
    BASIS_COLORS = {'signal_basis': ('#1f77b4', GOLD),           # blue  / gold
                    'difference_basis': ('#2ca02c', '#9467bd')}  # green / purple
    present = [k for k in ('signal_basis', 'difference_basis')
               if rec.get(k) is not None and (rec[k].get('analytic_recovery') is not None
                                              or rec[k].get('split_half_r') is not None)]
    multi = len(present) > 1
    # Marker/primary color follows the basis the operating point was CHOSEN on, so
    # the analytic markers (peak/chosen/trial-avg) match that trace and the fig6
    # objective (in compare mode this is the chosen, not merely the first, basis).
    _cb = (rec.get('chosen') or {}).get('basis')
    _prim = (f'{_cb}_basis' if _cb in ('signal', 'difference')
             and rec.get(f'{_cb}_basis') is not None
             else (present[0] if present else 'signal_basis'))
    mc_analytic, mc_splithalf = BASIS_COLORS[_prim]

    # analytic recovery SOLID (left axis), split-half r DASHED (right axis).
    for key, name in (('signal_basis', 'signal'), ('difference_basis', 'difference')):
        b = rec.get(key)
        if b is None:
            continue
        c_analytic, c_splithalf = BASIS_COLORS[key]
        a_lbl = f'{name}: analytic recovery' if multi else 'analytic recovery'
        s_lbl = f'{name}: split-half r' if multi else 'split-half r'
        if b.get('analytic_recovery') is not None and b.get('analytic_sv_frac') is not None:
            ax.plot(b['analytic_sv_frac'], b['analytic_recovery'], '-', color=c_analytic, lw=2,
                    label=a_lbl)
            yL.append(np.asarray(b['analytic_recovery']).ravel())
            if analytic_endpoint is None:
                _ar = np.asarray(b['analytic_recovery']).ravel()
                _asf = np.asarray(b['analytic_sv_frac']).ravel()
                if _ar.size:
                    analytic_endpoint, analytic_endpoint_x = float(_ar[-1]), float(_asf[-1])
                    prim_ar, prim_asf = _ar, _asf
        if b.get('split_half_r') is not None and b.get('sv_frac') is not None:
            ax_r.plot(b['sv_frac'], b['split_half_r'], '--', color=c_splithalf, lw=1.6, alpha=0.85,
                      label=s_lbl)
            yR.append(np.asarray(b['split_half_r']).ravel())

    # Max-tradeoff geometry: the chord from the prediction peak to the do-nothing
    # (trial-average) point, and the shaded gap between that chord and the analytic
    # recovery curve on the descending limb, where max-tradeoff picks the farthest point.
    if prim_ar is not None and prim_ar.size >= 3:
        kpk = int(np.argmax(prim_ar))
        # Peak of the analytic recovery curve (prediction peak): triangle.
        ax.scatter([prim_asf[kpk]], [prim_ar[kpk]], marker='^', s=130, color=mc_analytic,
                   edgecolor='k', linewidth=0.8, zorder=6, label='prediction peak (analytic)')
        yL.append(np.asarray([prim_ar[kpk]]))
        if kpk < prim_ar.size - 1 and prim_asf[-1] != prim_asf[kpk]:
            xs = prim_asf[kpk:]
            yc = prim_ar[kpk:]
            ychord = prim_ar[kpk] + (prim_ar[-1] - prim_ar[kpk]) * \
                (xs - prim_asf[kpk]) / (prim_asf[-1] - prim_asf[kpk])
            ax.fill_between(xs, ychord, yc, color='0.5', alpha=0.15, zorder=0)
            ax.plot(xs, ychord, ls=':', color='0.45', lw=1.0, zorder=1)

    # trial-average (do-nothing): split-half-colored box on the split-half (right)
    # axis + analytic-colored box on the analytic (left) curve. Wiener stays right.
    ta = rec.get('trial_average')
    if ta is not None and ta.get('split_half_r') is not None:
        ax_r.scatter([ta['sv_frac']], [ta['split_half_r']], marker='s', s=70, color=mc_splithalf,
                     edgecolor='k', linewidth=0.6, zorder=5, label='trial-avg (split-half)')
        yR.append(np.asarray([ta['split_half_r']]))
    # Analytic trial-avg box lives at the curve's full-retention endpoint, so it
    # is drawn from the analytic curve alone - independent of any split-half data
    # (present even when skip_split_half drops the split-half trial-avg marker).
    if analytic_endpoint is not None:
        ax.scatter([analytic_endpoint_x], [analytic_endpoint], marker='s', s=70, color=mc_analytic,
                   edgecolor='k', linewidth=0.6, zorder=5, label='trial-avg (analytic)')
        yL.append(np.asarray([analytic_endpoint]))
    w = rec.get('wiener')
    if w is not None:
        ax_r.scatter([w['sv_frac']], [w['split_half_r']], marker='D', s=70, color='limegreen',
                     edgecolor='k', linewidth=0.6, zorder=5, label='Wiener')
        yR.append(np.asarray([w['split_half_r']]))

    # Chosen operating point: analytic-colored star on the analytic (left) trajectory,
    # split-half-colored star on the split-half (right) trajectory.
    ch = rec.get('chosen')
    if ch is not None:
        if ch.get('split_half_r') is not None:
            ax_r.scatter([ch['sv_frac']], [ch['split_half_r']], marker='*', s=320, color=mc_splithalf,
                         edgecolor='k', linewidth=0.9, zorder=6, label='PSN chosen (split-half)')
            yR.append(np.asarray([ch['split_half_r']]))
        if ch.get('recovery') is not None:
            ax.scatter([ch['sv_frac']], [ch['recovery']], marker='*', s=320, color=mc_analytic,
                       edgecolor='k', linewidth=0.9, zorder=6, label='PSN chosen (analytic)')
            yL.append(np.asarray([ch['recovery']]))

    ax.set_xlabel('frac. signal var. retained')
    ax.set_ylabel('analytic recovery  (cumsum signal - noise/t)')
    ax_r.set_ylabel('split-half r  (TAvg vs Denoised)')

    # Y-limits per axis: set the top to top_mult x the peak-to-bottom range so the
    # data peak lands at 1/top_mult of the axis height, keeping the top clear for the
    # top-left legend (and, in max-tradeoff mode, the upper-right inset too). 3x in
    # max-tradeoff mode (needs room for the inset), 2x otherwise (legend only). When
    # values go negative (e.g. the do-nothing tail of the analytic-recovery curve on
    # noisy data) the lower limit follows the data instead of clipping at 0.
    is_mt = rec.get('criterion') == 'max-tradeoff'
    top_mult = 3.0 if is_mt else 2.0

    def _lim(yvals):
        v = np.concatenate(yvals) if yvals else None
        if v is None:
            return None
        v = v[np.isfinite(v)]
        if not v.size:
            return None
        vmax = max(float(np.max(v)), 1e-3)
        vmin = float(np.min(v))
        if vmin >= 0:
            return (0.0, top_mult * vmax)
        span = vmax - vmin
        bottom = vmin - 0.05 * span
        return (bottom, bottom + top_mult * (vmax - bottom))
    lL, lR = _lim(yL), _lim(yR)
    if lL:
        ax.set_ylim(*lL)
    if lR:
        ax_r.set_ylim(*lR)

    # Inverse-log x so the [0.9, 1] region (where curves and points crowd) fans out.
    _set_tail_invlog_xscale(ax)
    ax.grid(alpha=0.25)
    # Combined legend from both axes, ordered so the analytic (gold) entries form
    # one block and the split-half (gray) entries another, with gold box/star adjacent,
    # gray box/star adjacent.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax_r.get_legend_handles_labels()
    def _rank(lbl):
        L = lbl.lower()
        if 'prediction peak' in L:
            return 0.5 if 'analytic' in L else 9
        if 'chosen' in L:
            return 2 if 'analytic' in L else 14
        if 'trial-avg' in L:
            return 1 if 'analytic' in L else 13
        if 'tradeoff' in L:
            return 3
        if 'analytic recovery' in L:
            return 0
        if 'split-half r' in L:
            return 10
        if L == 'units':
            return 11
        if 'wiener' in L:
            return 12
        return 99
    pairs = sorted(zip(h1 + h2, l1 + l2), key=lambda hl: _rank(hl[1]))
    # Upper-left, two columns so the legend runs along the top and stays compact (the
    # ylim rule keeps the data in the lower half/third, so the upper area is clear).
    _legend(ax, [h for h, _ in pairs], [l for _, l in pairs],
            fontsize=6.5, ncol=2, columnspacing=1.0,
            handletextpad=0.5, loc='upper left', framealpha=0.85)

    # Inset (upper-right): linear-axis zoom of the max-tradeoff selection geometry.
    _draw_max_tradeoff_inset(ax, rec)


def _draw_max_tradeoff_inset(ax, rec):
    """Lower-left inset that zooms, on LINEAR axes, into the descending limb where
    the max-tradeoff criterion picks the operating point. Coordinates are
    normalized so the recovery peak sits at (0,1) and the trial-average
    (do-nothing) at (1,0); the peak->trial-average chord is then the anti-diagonal
    u+v=1, and max-tradeoff selects the curve point of greatest perpendicular
    distance from it (argmax of the shaded gap). argmax perpendicular distance is
    affine-invariant, so this clean square view marks the SAME operating point
    psn() chose while drawing a true right angle. Only rendered in max-tradeoff
    mode."""
    if rec.get('criterion') != 'max-tradeoff':
        return
    ch = rec.get('chosen')
    if not ch or ch.get('recovery') is None or ch.get('sv_frac') is None:
        return
    # Basis the operating point was chosen on (fallback: first present).
    key = f"{ch.get('basis')}_basis" if ch.get('basis') in ('signal', 'difference') else None
    if key is None or rec.get(key) is None:
        key = next((k for k in ('signal_basis', 'difference_basis')
                    if rec.get(k) is not None), None)
    if key is None:
        return
    b = rec[key]
    if b.get('analytic_sv_frac') is None or b.get('analytic_recovery') is None:
        return
    asf = np.asarray(b['analytic_sv_frac'], float)
    ar = np.asarray(b['analytic_recovery'], float)
    if asf.size < 3 or ar.size < 3:
        return
    kpk = int(np.argmax(ar))
    xpk, rpk, xend, rend = asf[kpk], ar[kpk], asf[-1], ar[-1]
    if kpk >= ar.size - 1 or xend == xpk or (rpk - rend) <= 1e-12 \
            or not np.isfinite(ch['recovery']):
        return

    # Normalized descending limb: u in [0,1] (0=peak, 1=trial-avg), v in [0,1] (1=peak).
    u = (asf[kpk:] - xpk) / (xend - xpk)
    v = (ar[kpk:] - rend) / (rpk - rend)
    u_ch = (ch['sv_frac'] - xpk) / (xend - xpk)
    v_ch = (ch['recovery'] - rend) / (rpk - rend)
    tt = (u_ch + v_ch - 1) / 2.0                      # foot of perpendicular on u+v=1
    foot = (u_ch - tt, v_ch - tt)

    mc = '#2ca02c' if key == 'difference_basis' else '#1f77b4'   # matches the panel trace
    perpc = '#d81a1a'

    # Top-right corner, pushed up and right into the whitespace the top headroom
    # opens up, with just enough gap above so the title does not hit the top frame.
    axi = ax.inset_axes([0.655, 0.52, 0.335, 0.335])
    axi.set_facecolor('white')
    axi.fill_between(u, v, 1 - u, color='0.5', alpha=0.12, lw=0)          # gap curve<->chord
    axi.plot([0, 1], [1, 0], '--', color='0.35', lw=1.0)                 # chord
    axi.plot(u, v, '-', color=mc, lw=1.6)                                # recovery curve
    axi.scatter([0], [1], marker='^', s=45, color=mc, edgecolor='k',
                linewidth=0.6, zorder=5)                                 # peak
    axi.scatter([1], [0], marker='s', s=40, color=mc, edgecolor='k',
                linewidth=0.6, zorder=5)                                 # trial-avg
    axi.plot([u_ch, foot[0]], [v_ch, foot[1]], '-', color=perpc, lw=1.4,
             zorder=6)                                                   # perpendicular
    axi.scatter([foot[0]], [foot[1]], s=18, facecolor='white',
                edgecolor=perpc, linewidth=1.0, zorder=6)                # foot
    axi.scatter([u_ch], [v_ch], marker='*', s=150, color=mc, edgecolor='k',
                linewidth=0.6, zorder=7)                                 # chosen

    axi.set_aspect('equal', adjustable='box')      # equal units -> the right angle reads true
    axi.set_xlim(-0.06, 1.10)
    axi.set_ylim(-0.06, 1.10)
    # Tick labels in the ACTUAL units: both axes are linear (affine) maps of them,
    # u -> frac. signal var. retained, v -> analytic recovery, so labelling the
    # fixed tick positions with the real values is exact.
    xt = np.array([0.0, 0.5, 1.0])
    axi.set_xticks(xt)
    axi.set_yticks(xt)
    axi.set_xticklabels(_fmt_ticks(xpk + xt * (xend - xpk)))
    axi.set_yticklabels(_fmt_ticks(rend + xt * (rpk - rend)))
    axi.tick_params(labelsize=6, length=2)
    axi.set_title('max-tradeoff (zoom)', fontsize=7, pad=2)
    axi.set_xlabel('frac. signal retained', fontsize=6.5, labelpad=1)
    axi.set_ylabel('analytic recovery', fontsize=6.5, labelpad=1)


def _fmt_ticks(vals):
    """Format tick values with the fewest decimals that keeps them all distinct
    (the descending limb can crowd near frac=1, so a fixed precision would collapse
    adjacent labels)."""
    vals = [float(v) for v in vals]
    for d in range(1, 9):
        s = [f'{v:.{d}f}' for v in vals]
        if len(set(s)) == len(s):
            return s
    return [f'{v:.8f}' for v in vals]



def plot_diagnostic_figures(data, results, test_data=None, figurepath=None, cmap=None,
                            split_half_metric='correlation'):
    """
    Generate diagnostic figures for PSN denoising results (NEW API).

    This visualization works with the new PSN API and results structure.

    Parameters:
    -----------
    data : ndarray
        Training data used for denoising, shape (nunits, nconds, ntrials)
    results : dict
        Results dictionary from psn function
    test_data : ndarray, optional
        Not used in current implementation (reserved for future use)
    figurepath : str, optional
        If specified, save figure to this path before displaying.
        The figure is saved at 150 dpi with tight bounding box.
    cmap : colormap, optional
        Colormap for input data, denoised data, and residual plots.
        Default: cmapsign4()
    split_half_metric : str, optional
        Metric for the split-half reliability plot.
        'correlation' (default) - Pearson r per unit.
        'mse' - mean squared error per unit.
    """
    # Set default colormap for data plots
    if cmap is None:
        cmap = cmapsign4()

    # Set random seed for reproducibility
    np.random.seed(42)

    # Increase font sizes globally for this figure
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })

    # Create a large figure with custom grid layout
    # Top row: 5 subplots (cSb, cNb, basis dims (half), eigenvalues (half), sig/noise var)
    # Rows 2-4: standard 4x4 grid
    fig = plt.figure(figsize=(27, 15))

    # Create GridSpec: 4 rows, 8 columns (to allow half-width subplots)
    # Increase spacing to prevent overlap
    # Use width_ratios to make first 2 columns slightly narrower (for objective plot with twin y-axis)
    gs = GridSpec(4, 8, figure=fig, hspace=0.5, wspace=0.42,
                  left=0.04, right=0.985, top=0.95, bottom=0.045,
                  width_ratios=[0.9, 0.9, 1, 1, 1, 1, 1, 1])

    # Feature 1: row 0 gets a 6th panel at the far right (recovery tradeoff),
    # keeping the original relative widths of the first five panels.
    gs_row0 = GridSpecFromSubplotSpec(1, 6, subplot_spec=gs[0, :],
                                      width_ratios=[2, 2, 1, 1, 2, 2], wspace=0.45)

    # Extract data dimensions
    nunits, nconds, ntrials = data.shape

    # Subsampling for large datasets (>500 units or >500 conditions)
    # Randomly select 100 units/conditions for certain plots, but keep full population for means
    n_subsample = 100

    # Unit subsampling
    subsample_units = nunits > 500
    if subsample_units:
        np.random.seed(42)  # For reproducibility
        subsample_idx = np.sort(np.random.choice(nunits, n_subsample, replace=False))
    else:
        subsample_idx = np.arange(nunits)

    # Condition subsampling (for trace plots)
    subsample_conds = nconds > 500
    if subsample_conds:
        np.random.seed(43)  # Different seed for conditions
        subsample_cond_idx = np.sort(np.random.choice(nconds, n_subsample, replace=False))
    else:
        subsample_cond_idx = np.arange(nconds)

    # Build suffix strings
    if subsample_units and subsample_conds:
        subsample_suffix = f'\n(randomly subsampling {n_subsample} units, {n_subsample} conditions)'
        subsample_suffix_units = f'\n(randomly subsampling {n_subsample} units)'
        subsample_suffix_conds = f'\n(randomly subsampling {n_subsample} conditions)'
        subsample_suffix_traces = f'\n(randomly subsampling {n_subsample} units, {n_subsample} conditions)'
    elif subsample_units:
        subsample_suffix = f'\n(randomly subsampling {n_subsample} units)'
        subsample_suffix_units = subsample_suffix
        subsample_suffix_conds = ''
        subsample_suffix_traces = subsample_suffix
    elif subsample_conds:
        subsample_suffix = ''
        subsample_suffix_units = ''
        subsample_suffix_conds = f'\n(randomly subsampling {n_subsample} conditions)'
        subsample_suffix_traces = subsample_suffix_conds
    else:
        subsample_suffix = ''
        subsample_suffix_units = ''
        subsample_suffix_conds = ''
        subsample_suffix_traces = ''

    # Get options if stored
    if 'opt_used' in results:
        opt = results['opt_used']
    else:
        opt = {}

    # Extract basis type description
    if 'basis' in opt:
        if isinstance(opt['basis'], str):
            basis_desc = opt['basis']
        else:
            basis_desc = f"custom [{opt['basis'].shape[0]}x{opt['basis'].shape[1]}]"
    else:
        basis_desc = 'unknown'

    # Extract threshold method
    threshold_method = opt.get('threshold_method', 'unknown')

    # Extract criterion
    criterion = opt.get('criterion', 'unknown')

    # Extract basis_ordering
    basis_ordering = opt.get('basis_ordering', 'eigenvalues')

    # Check for NaNs and compute average number of trials
    has_nans = np.any(np.isnan(data))
    if has_nans:
        validcnt = np.sum(~np.any(np.isnan(data), axis=0), axis=1)
        ntrials_avg = np.sum(validcnt[validcnt > 1]) / nconds
        if ntrials_avg < 1:
            ntrials_avg = 1
    else:
        ntrials_avg = ntrials

    # Create title (order: Basis, Criterion, Method to match API)
    if basis_desc == 'wiener':
        # Full-rank Wiener bypasses criterion/threshold - show simplified title
        data_str = (f'{nunits} units × {nconds} conditions × {ntrials} max trials (avg {ntrials_avg:.1f})'
                    if has_nans else f'{nunits} units × {nconds} conditions × {ntrials} trials')
        title_text = f'Data: {data_str}  |  Full-Rank Matrix Wiener Filter'
    elif has_nans:
        if opt.get('alpha') is not None:
            title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} max trials (avg {ntrials_avg:.1f})  |  Basis: {basis_desc}  |  Alpha: {opt["alpha"]}  |  Method: {threshold_method}'
        else:
            title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} max trials (avg {ntrials_avg:.1f})  |  Basis: {basis_desc}  |  Criterion: {criterion}  |  Method: {threshold_method}'
    else:
        if opt.get('alpha') is not None:
            title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} trials  |  Basis: {basis_desc}  |  Alpha: {opt["alpha"]}  |  Method: {threshold_method}'
        else:
            title_text = f'Data: {nunits} units × {nconds} conditions × {ntrials} trials  |  Basis: {basis_desc}  |  Criterion: {criterion}  |  Method: {threshold_method}'

    # Add threshold info if conservative mode or variance criterion is used
    threshold_info = []
    if 'allowable_thresholds' in opt and opt['allowable_thresholds'] is not None:
        allowable = opt['allowable_thresholds']
        if hasattr(allowable, '__len__'):
            if len(allowable) == 1:
                threshold_info.append(f"Forced threshold: {int(allowable[0])}")
            else:
                threshold_info.append(f"Allowable thresholds: {list(allowable)}")
        else:
            threshold_info.append(f"Forced threshold: {int(allowable)}")
    if opt.get('alpha') is not None:
        vt = opt.get('variance_threshold', 0.99)
        threshold_info.append(f"Variance target: {vt}")
    elif criterion in ['variance', 'variance_eigenvalues']:
        vt = opt.get('variance_threshold', 0.99)
        threshold_info.append(f"Variance threshold: {vt}")

    if threshold_info:
        title_text += '  |  ' + ', '.join(threshold_info)

    plt.suptitle(title_text, fontsize=14, fontweight='bold')

    # Get trial-averaged and denoised data (use nanmean for NaN data)
    if has_nans:
        trial_avg = np.nanmean(data, axis=2)
    else:
        trial_avg = np.mean(data, axis=2)

    denoised = results['denoiseddata']
    noise = trial_avg - denoised

    # =========================================================================
    # Plot 1: Basis source matrix (signal covariance or basis-specific)
    # =========================================================================
    ax1 = fig.add_subplot(gs_row0[0, 0])  # Row 0 panel 1
    if 'gsn_result' in results and 'cSb' in results['gsn_result']:
        cSb = results['gsn_result']['cSb']
        cNb = results['gsn_result']['cNb']

        # Determine which matrix based on basis type
        basis_type = opt.get('basis', 'signal')
        if isinstance(basis_type, str):
            if basis_type == 'difference':
                plot_matrix_1 = cSb - cNb / ntrials_avg
                plot_title = f'cSb - cNb/{ntrials_avg:.1f} (difference)'
            elif basis_type == 'noise':
                plot_matrix_1 = cNb
                plot_title = 'Noise Covariance (cNb)'
            elif basis_type == 'pca':
                trial_avg_demeaned = trial_avg - results['unit_means'][:, np.newaxis]
                plot_matrix_1 = np.cov(trial_avg_demeaned)
                plot_title = 'Trial-Avg Data Covariance'
            else:  # 'signal' or default
                plot_matrix_1 = cSb
                plot_title = 'Signal Covariance (cSb)'
        else:
            plot_matrix_1 = cSb
            plot_title = 'Signal Covariance (cSb)'

        # Subsample huge matrices before imshow / percentile (cheap at large N).
        plot_matrix_1, _ext1 = _subsample_for_imshow(plot_matrix_1)

        # Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
        if has_nans:
            data_absmax = np.nanpercentile(np.abs(plot_matrix_1), 99)
        else:
            data_absmax = np.percentile(np.abs(plot_matrix_1), 99)

        if data_absmax > 0:
            clim_1 = [-data_absmax, data_absmax]
        else:
            clim_1 = [-1, 1]

        im1 = ax1.imshow(plot_matrix_1, vmin=clim_1[0], vmax=clim_1[1], cmap=redblue(), aspect='equal', extent=_ext1)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.02)
        ax1.set_title(plot_title)
        ax1.set_xlabel('Units')
        ax1.set_ylabel('Units')
        ax1.locator_params(axis='both', nbins=4)   # fewer, non-overlapping ticks
        ax1.tick_params(labelsize=8)
    else:
        ax1.text(0.5, 0.5, 'Covariance\nNot Available',
                ha='center', va='center', transform=ax1.transAxes)

    # =========================================================================
    # Plot 2: Noise Covariance (cNb)
    # =========================================================================
    ax2 = fig.add_subplot(gs_row0[0, 1])  # Row 0 panel 2
    if 'gsn_result' in results and 'cNb' in results['gsn_result']:
        cNb = results['gsn_result']['cNb']

        # Subsample huge matrices before imshow / percentile (cheap at large N).
        cNb, _ext2 = _subsample_for_imshow(cNb)

        # Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
        if has_nans:
            data_absmax_cNb = np.nanpercentile(np.abs(cNb), 99)
        else:
            data_absmax_cNb = np.percentile(np.abs(cNb), 99)

        if data_absmax_cNb > 0:
            clim_cNb = [-data_absmax_cNb, data_absmax_cNb]
        else:
            clim_cNb = [-1, 1]

        im2 = ax2.imshow(cNb, vmin=clim_cNb[0], vmax=clim_cNb[1], cmap=redblue(), aspect='equal', extent=_ext2)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02)
        ax2.set_title('Noise Covariance (cNb)')
        ax2.set_xlabel('Units')
        ax2.set_ylabel('Units')
        ax2.locator_params(axis='both', nbins=4)   # fewer, non-overlapping ticks
        ax2.tick_params(labelsize=8)
    else:
        ax2.text(0.5, 0.5, 'Noise Covariance\nNot Available',
                ha='center', va='center', transform=ax2.transAxes)

    # =========================================================================
    # Plot 3: Top 5 PCs as vertical line plots (half width)
    # =========================================================================
    ax3 = fig.add_subplot(gs_row0[0, 2])  # Row 0 panel 3 (half width)
    if 'fullbasis' in results:
        num_pcs = min(5, results['fullbasis'].shape[1])

        # Normalize each PC for visualization (0-indexed)
        y_units = np.arange(nunits)
        colors = plt.cm.tab10(np.linspace(0, 1, num_pcs))

        # Find max absolute loading across top 5 PCs for scaling
        max_loading = np.max(np.abs(results['fullbasis'][:, :num_pcs]))
        if max_loading > 0:
            scale_factor = 0.4 / max_loading  # Scale to fit within 0.4 x-units
        else:
            scale_factor = 1

        for pc in range(num_pcs):
            # Center each PC at position pc (0-indexed), with loadings as horizontal deviations
            x_vals = pc + results['fullbasis'][:, pc] * scale_factor
            ax3.plot(x_vals, y_units, linewidth=1.5, color=colors[pc])

            # Add vertical reference line at center
            ax3.axvline(x=pc, color='k', linestyle='--', linewidth=0.5)

        ax3.set_xlabel('Principal Component')
        ax3.set_ylabel('Units')
        ax3.set_title('Top 5 Basis Dims')
        ax3.set_xlim([-0.5, num_pcs - 0.5])
        ax3.set_ylim([0, nunits - 1])
        ax3.invert_yaxis()  # Flip y-axis to match heatmaps
        ax3.set_xticks(range(num_pcs))
        # Add eigenvalue labels under PC indices
        if 'basis_eigenvalues' in results and results['basis_eigenvalues'] is not None and len(results['basis_eigenvalues']) >= num_pcs:
            evals = results['basis_eigenvalues']
            def _lam(v):  # compact eigenvalue label so 5 fit in a half-panel
                av = abs(v)
                return f'{v:.0f}' if av >= 100 else (f'{v:.1f}' if av >= 1 else f'{v:.2f}')
            tick_labels = [f'{pc}\nλ={_lam(evals[pc])}' for pc in range(num_pcs)]
            ax3.set_xticklabels(tick_labels, fontsize=6)
        ax3.grid(True)
    else:
        ax3.text(0.5, 0.5, 'Basis\nNot Available',
                ha='center', va='center', transform=ax3.transAxes)

    # =========================================================================
    # Plot 4: Global dimension ranking (eigenvalues or signal variance) - half width
    # =========================================================================
    ax4 = fig.add_subplot(gs_row0[0, 3])  # Row 0 panel 4 (half width)

    # Determine if we should use log scale for x-axis (for large datasets)
    use_logscale = nunits > 50

    # Check if eigenvalues were used for ranking (and are available)
    use_eigenvalues = (basis_ordering == 'eigenvalues' and
                      'basis_eigenvalues' in results and
                      results['basis_eigenvalues'] is not None and
                      len(results['basis_eigenvalues']) > 0)

    # Check if prediction ordering was used
    use_prediction_ordering = (basis_ordering == 'prediction')

    # Flag for full-rank Wiener (changes how threshold lines are labeled)
    is_fullrank_wiener = (basis_desc == 'wiener')

    if use_eigenvalues:
        # Show eigenvalues (SORTED - what was actually used for ranking)
        evals = results['basis_eigenvalues']  # Already sorted in descending order
        # For log scale, shift x by 1 so dimensions go 1, 2, 3... (avoids log(0))
        if use_logscale:
            x_vals = np.arange(len(evals)) + 1
        else:
            x_vals = np.arange(len(evals))  # 0-indexed
        ax4.plot(x_vals, evals, linewidth=1.5, color=[0.5, 0, 0.5], label='$\\lambda_k(\\Sigma_S)$')

        # Add threshold indicators (only if threshold > 0)
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                if is_fullrank_wiener:
                    thresh_label = f'$\\mathrm{{tr}}(D) = {best_t:.1f}$'
                else:
                    thresh_label = f'Threshold $K = {int(best_t)}$'
                ax4.axvline(x=best_t, color='r', linestyle='--', linewidth=2, label=thresh_label)
            elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                mean_thresh = np.mean(best_t)
                thresh_label = f'Mean threshold $= {mean_thresh:.1f}$'
                ax4.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2, label=thresh_label)

        ax4.set_xlabel('Dimension $k$ (signal eigenbasis)')
        ax4.set_ylabel('Eigenvalue')
        if is_fullrank_wiener:
            ax4.set_title('$\\Sigma_S$ Eigenvalues (signal basis)')
        else:
            ax4.set_title('Basis Eigenvalues')
        _legend(ax4, loc='best', fontsize=7)
        ax4.grid(True)
        if use_logscale:
            _set_headtail_log_xscale(ax4, len(evals), rotation=45)
        else:
            ax4.set_xlim([-0.5, len(evals) - 0.5])

    elif use_prediction_ordering and 'signalvar' in results and 'noisevar' in results:
        # Show prediction ordering criterion (signal - noise/ntrials) and signal variance
        signal_vars = results['signalvar']
        noise_vars = results['noisevar']
        prediction_obj = signal_vars - noise_vars / ntrials_avg

        if use_logscale:
            x_vals = np.arange(len(signal_vars)) + 1
        else:
            x_vals = np.arange(len(signal_vars))

        # Plot both signal variance and prediction objective
        ax4.plot(x_vals, signal_vars, linewidth=1.5, color='blue', label='Signal Var')
        ax4.plot(x_vals, prediction_obj, linewidth=1.5, color=[0.5, 0, 0.5], label='SigVar - NoiseVar/ntrials')

        # Add threshold indicators (only if threshold > 0)
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                ax4.axvline(x=best_t, color='r', linestyle='--', linewidth=2)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(best_t * 1.05 if use_logscale else best_t + 0.5, y_pos, f'Threshold = {int(best_t)}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')
            elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                mean_thresh = np.mean(best_t)
                ax4.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(mean_thresh * 1.05 if use_logscale else mean_thresh + 0.5, y_pos, f'Mean Threshold = {mean_thresh:.1f}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')

        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Variance')
        ax4.set_title('Ordering Criterion')
        _legend(ax4, loc='best', fontsize=7)
        ax4.grid(True)
        if use_logscale:
            _set_headtail_log_xscale(ax4, len(signal_vars), rotation=45)
        else:
            ax4.set_xlim([-0.5, len(signal_vars) - 0.5])

    elif 'signalvar' in results:
        # Show signal variance (SORTED - what was actually used for ranking)
        signal_vars = results['signalvar']  # Already sorted in descending order
        # For log scale, shift x by 1 so dimensions go 1, 2, 3... (avoids log(0))
        if use_logscale:
            x_vals = np.arange(len(signal_vars)) + 1
        else:
            x_vals = np.arange(len(signal_vars))  # 0-indexed
        ax4.plot(x_vals, signal_vars, linewidth=1.5, color='blue')

        # Add threshold indicators (only if threshold > 0)
        if 'best_threshold' in results:
            best_t = results['best_threshold']
            if np.isscalar(best_t) and best_t > 0:
                ax4.axvline(x=best_t, color='r', linestyle='--', linewidth=2)
                # Add rotated text annotation (top of text on right side of line)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(best_t * 1.05 if use_logscale else best_t + 0.5, y_pos, f'Threshold = {int(best_t)}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')
            elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                mean_thresh = np.mean(best_t)
                ax4.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2)
                # Add rotated text annotation (top of text on right side of line)
                ylims = ax4.get_ylim()
                y_pos = ylims[0] + 0.7 * (ylims[1] - ylims[0])
                ax4.text(mean_thresh * 1.05 if use_logscale else mean_thresh + 0.5, y_pos, f'Mean Threshold = {mean_thresh:.1f}',
                        color='r', fontsize=9, rotation=90,
                        ha='left', va='top')

        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Signal Var')
        ax4.set_title('Signal Variance')
        ax4.grid(True)
        if use_logscale:
            _set_headtail_log_xscale(ax4, len(signal_vars), rotation=45)
        else:
            ax4.set_xlim([-0.5, len(signal_vars) - 0.5])
    else:
        ax4.text(0.5, 0.5, 'Ranking Info\nNot Available',
                ha='center', va='center', transform=ax4.transAxes)

    # =========================================================================
    # Plot 5: Signal vs Noise variance
    # =========================================================================
    ax5 = fig.add_subplot(gs_row0[0, 4])  # Row 0 panel 5
    if 'signalvar' in results and 'noisevar' in results:
        if not isinstance(results['signalvar'], (list, tuple)):
            # Global or averaged
            sv = results['signalvar']
            nv = results['noisevar']

            # Left y-axis for variance
            ax5_left = ax5
            ax5_left.set_ylabel('Variance', color='tab:blue')

            # For log scale, shift x by 1 so dimensions go 1, 2, 3... (avoids log(0))
            if use_logscale:
                x_vals = np.arange(len(sv)) + 1
            else:
                x_vals = np.arange(len(sv))  # 0-indexed
            line1 = ax5_left.plot(x_vals, sv, '-', linewidth=1.5, color='blue', label='Signal var')
            line2 = ax5_left.plot(x_vals, nv, '-', linewidth=1.5, color=[1, 0.5, 0], label='Noise var')
            line2b = ax5_left.plot(x_vals, nv / ntrials_avg, '-', linewidth=1.5, color=[1, 0.85, 0.6], label=f'Noise var / {ntrials_avg:.1f} trials')

            ax5_left.tick_params(axis='y', labelcolor='tab:blue')

            # Right y-axis for NCSNR
            ax5_right = ax5_left.twinx()
            # Rectify negative signal variance to 0 (cSb need not be strictly PSD).
            # NCSNR is undefined where the projected noise variance underflows to
            # ~0: a signal-basis direction can land in a null direction of cNb, so
            # noisevar = u'·cNb·u ≈ 0 while signalvar is appreciable. Dividing by the
            # bare +eps floor (sqrt(eps) ≈ 1.5e-8) then explodes the trace to ~1e7.
            # Floor relative to the data and mask those dims so the line shows a gap
            # instead of a spurious spike.
            nv_arr = np.asarray(nv, dtype=float)
            noise_floor = max(np.finfo(float).eps, 1e-8 * np.nanmax(nv_arr))
            ncsnr_trace = np.where(
                nv_arr >= noise_floor,
                np.sqrt(np.maximum(sv, 0)) / np.sqrt(np.maximum(nv_arr, noise_floor)),
                np.nan,
            )
            line3 = ax5_right.plot(x_vals, ncsnr_trace, '-', linewidth=1.5, color='magenta', label='NCSNR')

            ax5_right.set_ylabel('NCSNR', color='magenta')
            ax5_right.tick_params(axis='y', labelcolor='magenta')

            # Add threshold (on left axis, only if > 0)
            line_thresh = None
            if 'best_threshold' in results:
                best_t = results['best_threshold']
                if np.isscalar(best_t) and best_t > 0:
                    if is_fullrank_wiener:
                        thresh_label = f'$\\mathrm{{tr}}(D) = {best_t:.1f}$  where $D = \\Sigma_S(\\Sigma_S + \\Sigma_N/t)^{{-1}}$'
                    else:
                        thresh_label = f'Threshold $K = {int(best_t)}$'
                    line_thresh = ax5_left.axvline(x=best_t, color='r', linestyle='--', linewidth=2, label=thresh_label)
                elif hasattr(best_t, '__len__') and np.mean(best_t) > 0:
                    mean_thresh = np.mean(best_t)
                    thresh_label = f'Mean threshold $= {mean_thresh:.1f}$'
                    line_thresh = ax5_left.axvline(x=mean_thresh, color='r', linestyle='--', linewidth=2, label=thresh_label)

            ax5_left.set_xlabel('Dimension $k$ (signal eigenbasis)' if is_fullrank_wiener else 'Dimension')
            ax5_left.set_title('Signal and Noise Variance')

            # Combine legends
            lines = line1 + line2 + line2b + line3
            if line_thresh is not None:
                lines = lines + [line_thresh]
            labels = [l.get_label() for l in lines]
            _legend(ax5_left, lines, labels, loc='best', fontsize=7)
            ax5_left.grid(True)

            if use_logscale:
                _set_headtail_log_xscale(ax5_left, len(sv), rotation=45)
            else:
                ax5_left.set_xlim([-0.5, len(sv) * 1.02])  # Push x-axis limit beyond final dimension
        else:
            ax5.text(0.5, 0.5, 'Per-Unit Variance\n(Averaged across units)',
                    ha='center', va='center', transform=ax5.transAxes)
    else:
        ax5.text(0.5, 0.5, 'Variance Info\nNot Available',
                ha='center', va='center', transform=ax5.transAxes)

    # =========================================================================
    # Plot 5b (NEW, Feature 1): recovery / bias-variance tradeoff (first panel of
    # row 1, next to the three data matrices).
    # =========================================================================
    ax_rec = fig.add_subplot(gs[1, 0:2])  # Row 1, columns 0-1
    # The data is precomputed in psn() (results['recovery_tradeoff']), regardless
    # of wantfig - the figure just renders it.
    if results.get('recovery_tradeoff') is not None:
        _plot_recovery_tradeoff(ax_rec, results['recovery_tradeoff'])
    else:
        ax_rec.set_title('Recovery vs. signal retained')
        ax_rec.text(0.5, 0.5, 'no GSN covariances', ha='center', va='center',
                    transform=ax_rec.transAxes)

    # =========================================================================
    # Plot 6: Objective function (cumulative signal - noise/t).
    # Moved to the far right of row 0 (swapped with the recovery-tradeoff panel).
    # =========================================================================
    ax6 = fig.add_subplot(gs_row0[0, 5])  # Row 0, panel 6 (far right)

    # Objective curve color, matched to the recovery panel: blue for the signal
    # basis, green for the difference basis (default green otherwise).
    _obj_basis = (results.get('threshold_selection') or {}).get('basis')
    if _obj_basis is None and isinstance(opt.get('basis'), str):
        _obj_basis = opt.get('basis')
    obj_color = {'signal': '#1f77b4', 'difference': '#2ca02c'}.get(_obj_basis, [0.3, 0.7, 0.3])

    # Check if Wiener mode (includes all Wiener-family denoisers)
    if 'wiener_weights' in results:
        # Wiener mode: show Wiener weights and cumulative objective on dual y-axes
        wiener_weights = results['wiener_weights']
        n_weights = len(wiener_weights)

        # For log scale, use x=0.5 for dimension 0, then 1, 2, 3...
        zero_placeholder = 0.5
        if use_logscale:
            x_dims = np.arange(1, n_weights + 1)
        else:
            x_dims = np.arange(n_weights)

        # Left y-axis: Wiener weights (line plot)
        line_weights, = ax6.plot(x_dims, wiener_weights, linewidth=2, color=[0.3, 0.5, 0.8],
                                  label='Wiener weights (w_k)')

        # Add a horizontal line at w=0.5 for reference
        ax6.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

        # Mark the effective dimensionality (sum of weights)
        effective_dims = np.sum(wiener_weights)
        ax6.axvline(x=effective_dims, color='red', linestyle='--', linewidth=2)

        ax6.set_ylabel('Wiener Weight (w_k)', color=[0.3, 0.5, 0.8])
        ax6.tick_params(axis='y', labelcolor=[0.3, 0.5, 0.8])
        ax6.set_ylim([0, 1.05])

        # Right y-axis: Cumulative objective (SignalVar - NoiseVar/ntrials)
        ax6_right = ax6.twinx()
        if 'objective' in results:
            obj = results['objective']
            # Objective has n+1 values (0 to n dims), align with weights
            if use_logscale:
                x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(obj))])
            else:
                x_obj = np.arange(len(obj))
            line_obj, = ax6_right.plot(x_obj, obj, linewidth=2, color=obj_color,
                                        label='analytic recovery')
            ax6_right.set_ylabel('analytic recovery  (cumsum signal - noise/t)', color=obj_color)
            ax6_right.tick_params(axis='y', labelcolor=obj_color)

            # Peak of analytic recovery (triangle) + do-nothing box at full retention.
            max_idx = int(np.argmax(obj))
            ndims = len(obj) - 1
            h_peak, = ax6_right.plot(x_obj[max_idx], obj[max_idx], '^', markersize=11,
                                     color=obj_color, markeredgecolor='black',
                                     markeredgewidth=0.8, zorder=10,
                                     label=f'prediction peak (K={max_idx})')
            h_box, = ax6_right.plot(x_obj[ndims], obj[ndims], 's', markersize=10,
                                    color=obj_color, markeredgecolor='black',
                                    markeredgewidth=0.8, zorder=10,
                                    label=f'trial-avg / do-nothing (K={ndims})')

            # Combined legend
            lines = [line_weights, line_obj, h_peak, h_box]
            labels = [l.get_label() for l in lines]
            _legend(ax6, lines, labels, loc='best', fontsize=8)

        ax6.set_xlabel('Dimension')
        ax6.set_title(f'Wiener weights & analytic recovery (eff. dims = {effective_dims:.1f})')
        ax6.grid(True, alpha=0.3)

        # Use log scale for x-axis if many dimensions
        if use_logscale:
            _set_headtail_log_xscale(ax6, n_weights, zero_at=zero_placeholder)
        else:
            ax6.set_xlim([-0.5, n_weights - 0.5])

    elif 'objective' in results and results['objective'] is not None:
        obj = results['objective']

        # For log scale, we need to handle x=0 specially
        # Use x=0.5 for the zero-dimension case, then 1, 2, 3... for rest
        zero_placeholder = 0.5

        # Check if unit-specific objectives are available
        if 'unit_objectives' in results and results['unit_objectives']:
            # Unit-specific mode: use dual y-axes
            # Left axis: unit curves (gray) - use subsampling if needed
            ax6_left = ax6
            h_units = None
            for u in subsample_idx:
                u_obj = results['unit_objectives'][u]
                if use_logscale:
                    # x=0 -> zero_placeholder, x=1 -> 1, x=2 -> 2, etc.
                    x_unit = np.concatenate([[zero_placeholder], np.arange(1, len(u_obj))])
                else:
                    x_unit = np.arange(len(u_obj))
                line, = ax6_left.plot(x_unit, u_obj, linewidth=0.5,
                        color=[0.5, 0.5, 0.5], alpha=0.3)
                if h_units is None:
                    h_units = line

            # Mark each unit's chosen threshold (on left axis) - use subsampling
            # Color by unit_groups if available
            if 'best_threshold' in results:
                best_t = results['best_threshold']
                if hasattr(best_t, '__len__'):
                    # Check if unit_groups are available for coloring
                    unit_groups = opt.get('unit_groups', None)
                    if unit_groups is not None:
                        unique_groups = np.unique(unit_groups)
                        n_groups = len(unique_groups)
                        # Use hsv colormap which handles arbitrary numbers of groups
                        group_colors = plt.cm.hsv(np.linspace(0, 0.9, n_groups))  # 0.9 to avoid wrapping back to red
                        # Create mapping from group to color
                        group_to_color = {g: group_colors[i] for i, g in enumerate(unique_groups)}

                    x_thresh = []
                    y_thresh = []
                    c_thresh = []  # colors for each point
                    for u in subsample_idx:
                        if u < len(best_t):
                            u_obj = results['unit_objectives'][u]
                            k_u = int(best_t[u])
                            if k_u >= 0 and k_u < len(u_obj):
                                if use_logscale:
                                    if k_u == 0:
                                        x_thresh.append(zero_placeholder)
                                    else:
                                        x_thresh.append(k_u)
                                else:
                                    x_thresh.append(k_u)
                                y_thresh.append(u_obj[k_u])
                                # Get color based on unit group
                                if unit_groups is not None and u < len(unit_groups):
                                    c_thresh.append(group_to_color[unit_groups[u]])
                                else:
                                    c_thresh.append([1, 0.3, 0.3, 0.6])  # default red
                    if x_thresh:
                        if unit_groups is not None:
                            ax6_left.scatter(x_thresh, y_thresh, s=20, c=c_thresh,
                                       alpha=0.6, zorder=5)
                        else:
                            ax6_left.scatter(x_thresh, y_thresh, s=20, color=[1, 0.3, 0.3],
                                       alpha=0.6, zorder=5)

            ax6_left.set_ylabel('unit analytic recovery\n(cumsum signal - noise/t)', color=[0.4, 0.4, 0.4])
            ax6_left.tick_params(axis='y', labelcolor=[0.4, 0.4, 0.4])

            # Right axis: population sum (green) - FULL population
            ax6_right = ax6.twinx()
            if use_logscale:
                x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(obj))])
            else:
                x_obj = np.arange(len(obj))
            h_sum, = ax6_right.plot(x_obj, obj, linewidth=2, color=obj_color)
            ax6_right.set_ylabel('analytic recovery (population)', color=obj_color)
            ax6_right.tick_params(axis='y', labelcolor=obj_color)

            # Peak of population analytic recovery (triangle) + do-nothing box at
            # full retention. Per-unit chosen thresholds are the scatter dots above.
            max_idx = int(np.argmax(obj))
            ndims = len(obj) - 1
            h_peak, = ax6_right.plot(x_obj[max_idx], obj[max_idx], '^', markersize=11,
                                     color=obj_color, markeredgecolor='black',
                                     markeredgewidth=0.8, zorder=10)
            h_box, = ax6_right.plot(x_obj[ndims], obj[ndims], 's', markersize=10,
                                    color=obj_color, markeredgecolor='black',
                                    markeredgewidth=0.8, zorder=10)

            alpha_info = results.get('alpha_info')
            legend_handles = [h_units, h_sum, h_peak, h_box]
            legend_labels = ['units', 'analytic recovery (population)',
                             f'prediction peak (K={max_idx})',
                             f'trial-avg / do-nothing (K={ndims})']

            # Alpha-specific: shade the interpolation range (prediction peak -> chosen).
            if alpha_info is not None:
                k_pred = alpha_info['k_pred']
                k_var = alpha_info['k_var']
                x_lo = (zero_placeholder if k_pred == 0 else k_pred) if use_logscale else k_pred
                x_hi = (zero_placeholder if k_var == 0 else k_var) if use_logscale else k_var
                if x_lo != x_hi:
                    ax6.axvspan(x_lo, x_hi, alpha=0.08, color='blue')

            _legend(ax6_left, legend_handles, legend_labels, loc='best', fontsize=7)

            if alpha_info is not None:
                ax6.set_title(f'Analytic recovery vs. dimensions (alpha={alpha_info["alpha"]}){subsample_suffix_units}')
            else:
                ax6.set_title(f'Analytic recovery vs. dimensions (unit-specific){subsample_suffix_units}')
        else:
            # Global mode: single curve
            if use_logscale:
                x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(obj))])
            else:
                x_obj = np.arange(len(obj))
            ax6.plot(x_obj, obj, linewidth=1.5, color=obj_color)

            alpha_info = results.get('alpha_info')
            ndims = len(obj) - 1

            def _xpos(idx):
                # Map a dimension index to its x position (log placeholder at 0).
                if use_logscale:
                    return zero_placeholder if idx == 0 else idx
                return idx

            # Peak of the analytic recovery curve (prediction peak): triangle.
            max_idx = int(np.argmax(obj))
            ax6.plot(_xpos(max_idx), obj[max_idx], '^', markersize=11,
                     color=obj_color, markeredgecolor='black',
                     markeredgewidth=0.8, zorder=10,
                     label=f'prediction peak (K={max_idx})')

            # Chosen threshold (may be constrained, so not necessarily the peak): star.
            if 'best_threshold' in results and np.isscalar(results['best_threshold']):
                k = int(results['best_threshold'])
                if k >= 0 and k < len(obj):
                    ax6.plot(_xpos(k), obj[k], '*', markersize=16,
                             color=obj_color, markeredgecolor='black',
                             markeredgewidth=0.9, zorder=11,
                             label=f'PSN chosen (K={k})')

            # Trial-average / do-nothing (keep every dimension): box at full retention.
            ax6.plot(_xpos(ndims), obj[ndims], 's', markersize=10,
                     color=obj_color, markeredgecolor='black',
                     markeredgewidth=0.8, zorder=10,
                     label=f'trial-avg / do-nothing (K={ndims})')

            # Alpha-specific: shade the interpolation range (prediction peak -> chosen).
            if alpha_info is not None:
                k_pred = alpha_info['k_pred']
                k_var = alpha_info['k_var']
                x_lo, x_hi = _xpos(k_pred), _xpos(k_var)
                if x_lo != x_hi:
                    ax6.axvspan(x_lo, x_hi, alpha=0.08, color='blue')

            _legend(ax6, loc='best', fontsize=7)

            if alpha_info is not None:
                ax6.set_title(f'Analytic recovery vs. dimensions (alpha={alpha_info["alpha"]})')
            else:
                ax6.set_title('Analytic recovery vs. dimensions')

        ax6.set_xlabel('Number of Dimensions')

        # Set ylabel based on criterion (only for global mode - unit-specific mode already set ylabels)
        if not ('unit_objectives' in results and results['unit_objectives']):
            if criterion == 'variance':
                ax6.set_ylabel('cumulative signal variance')
            else:
                ax6.set_ylabel('analytic recovery  (cumsum signal - noise/t)')

        ax6.grid(True)

        # Apply log scale and fix tick labels if needed
        if use_logscale:
            _set_headtail_log_xscale(ax6, len(obj) - 1, zero_at=zero_placeholder)
    elif basis_desc == 'wiener' and 'signalvar' in results and 'noisevar' in results:
        # Full-rank Wiener: show implied per-dimension weights and cumulative objective
        signal_proj = results['signalvar']
        noise_proj = results['noisevar']
        n_dims = len(signal_proj)

        # Compute implied Wiener weights: w_k = s_k / (s_k + n_k/t)
        denom = signal_proj + noise_proj / ntrials_avg
        wiener_weights = np.zeros_like(signal_proj)
        valid = denom > 0
        wiener_weights[valid] = signal_proj[valid] / denom[valid]
        wiener_weights = np.clip(wiener_weights, 0.0, 1.0)

        # Compute cumulative prediction objective: cumsum(s_k - n_k/t)
        prediction_obj = signal_proj - noise_proj / ntrials_avg
        cumsum_obj = np.concatenate([[0], np.cumsum(prediction_obj)])

        if use_logscale:
            x_dims = np.arange(1, n_dims + 1)
            zero_placeholder = 0.5
        else:
            x_dims = np.arange(n_dims)

        # Left y-axis: Wiener weights
        line_weights, = ax6.plot(x_dims, wiener_weights, linewidth=2, color=[0.3, 0.5, 0.8],
                                  label='Wiener weight $w_k = s_k / (s_k + n_k/t)$')
        ax6.axhline(y=0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        ax6.set_ylabel('Wiener Weight ($w_k$)', color=[0.3, 0.5, 0.8])
        ax6.tick_params(axis='y', labelcolor=[0.3, 0.5, 0.8])
        ax6.set_ylim([0, 1.05])

        # Right y-axis: Cumulative prediction objective
        ax6_right = ax6.twinx()
        if use_logscale:
            x_obj = np.concatenate([[zero_placeholder], np.arange(1, len(cumsum_obj))])
        else:
            x_obj = np.arange(len(cumsum_obj))
        line_obj, = ax6_right.plot(x_obj, cumsum_obj, linewidth=2, color=obj_color,
                                    label='Cumsum($s_k - n_k/t$)')
        ax6_right.set_ylabel('Cumulative $s_k - n_k/t$', color=obj_color)
        ax6_right.tick_params(axis='y', labelcolor=obj_color)

        # Star at max of global cumsum objective
        max_idx = np.argmax(cumsum_obj)
        h_star, = ax6_right.plot(x_obj[max_idx], cumsum_obj[max_idx], '*', markersize=14,
                                  color=obj_color, markeredgecolor='black',
                                  markeredgewidth=0.8, zorder=10,
                                  label=f'Max objective (K={max_idx})')

        # Mark effective dimensionality
        effective_dims = results.get('best_threshold', np.sum(wiener_weights))
        ax6.axvline(x=effective_dims if not use_logscale else max(effective_dims, 0.5),
                    color='red', linestyle='--', linewidth=2, label=f'Eff. dims = {effective_dims:.1f}')

        # Legend
        lines = [line_weights, line_obj, h_star]
        labels = [l.get_label() for l in lines]
        _legend(ax6, lines, labels, loc='best', fontsize=8)

        ax6.set_xlabel('Dimension (cSb eigenbasis)')
        ax6.set_title(f'Implied Wiener Weights in Signal Basis (eff. dims = {effective_dims:.1f})')
        ax6.grid(True, alpha=0.3)

        if use_logscale:
            _set_headtail_log_xscale(ax6, n_dims, zero_at=zero_placeholder)
        else:
            ax6.set_xlim([-0.5, n_dims - 0.5])

    else:
        ax6.text(0.5, 0.5, 'Objective\nNot Available',
                ha='center', va='center', transform=ax6.transAxes)

    # =========================================================================
    # Plot 7-9: Raw, Denoised, Noise
    # =========================================================================

    # Compute shared colorbar limits across all three plots (mean-centered)
    all_data_789 = np.concatenate([trial_avg.ravel(), denoised.ravel(), noise.ravel()])
    if has_nans:
        shared_mean = np.nanmean(all_data_789)
        shared_std = np.nanstd(all_data_789)
    else:
        shared_mean = np.mean(all_data_789)
        shared_std = np.std(all_data_789)
    if shared_std > 0:
        clim_shared = [shared_mean - 2*shared_std, shared_mean + 2*shared_std]
    else:
        clim_shared = [shared_mean - 1, shared_mean + 1]

    # Plots 7-9 (the three Units x Conditions imshows) live in their own tight
    # sub-grid so they pack together, while the gap from the objective panel
    # (its right twin-axis) stays generous via the main GridSpec wspace.
    _gs_imrow = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[1, 2:8], wspace=0.12)

    # Plot 7: Raw trial-averaged data (subsample the units axis at large N)
    ax7 = fig.add_subplot(_gs_imrow[0, 0])
    _ta_s, _ext_ta = _subsample_for_imshow(trial_avg)
    im7 = ax7.imshow(_ta_s, vmin=clim_shared[0], vmax=clim_shared[1], cmap=cmap, aspect='auto', interpolation='none', extent=_ext_ta)
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.02)
    title_7 = 'Input Data (trial-averaged, with NaNs)' if has_nans else 'Input Data (trial-averaged)'
    ax7.set_title(title_7)
    ax7.set_xlabel('Conditions')
    # No 'Units' ylabel word here: the y-ticks already show the unit scale, and
    # the word collides with Plot 6's right (twin) y-axis label.
    ax7.tick_params(axis='y', labelsize=8)

    # Plot 8: Denoised data
    ax8 = fig.add_subplot(_gs_imrow[0, 1])
    _dn_s, _ext_dn = _subsample_for_imshow(denoised)
    im8 = ax8.imshow(_dn_s, vmin=clim_shared[0], vmax=clim_shared[1], cmap=cmap, aspect='auto', interpolation='none', extent=_ext_dn)
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.02)
    ax8.set_title('PSN Denoised Data')
    ax8.set_xlabel('Conditions')
    ax8.set_yticklabels([])   # shares the 'Units' axis with Plot 7; drop redundant
                              # y labels so Plot 7's colorbar ticks don't collide.

    # Plot 9: Noise (residual)
    ax9 = fig.add_subplot(_gs_imrow[0, 2])
    _no_s, _ext_no = _subsample_for_imshow(noise)
    im9 = ax9.imshow(_no_s, vmin=clim_shared[0], vmax=clim_shared[1], cmap=cmap, aspect='auto', interpolation='none', extent=_ext_no)
    plt.colorbar(im9, ax=ax9, fraction=0.046, pad=0.02)
    title_9 = 'Residual (Noise, with NaNs)' if has_nans else 'Residual (Noise)'
    ax9.set_title(title_9)
    ax9.set_xlabel('Conditions')
    ax9.set_yticklabels([])

    # =========================================================================
    # Plot 10: Denoiser matrix
    # =========================================================================
    ax10 = fig.add_subplot(gs[2, 0:2])  # Row 2, columns 0-1
    denoiser = results['denoiser']
    if basis_desc == 'wiener' and 'wiener_matrix' in results:
        plot_matrix_10 = results['wiener_matrix']
        title_10 = 'Wiener Filter Matrix'
    else:
        plot_matrix_10 = denoiser
        title_10 = 'Denoiser Matrix'
    # Subsample huge matrices before imshow / percentile (cheap at large N).
    plot_matrix_10, _ext10 = _subsample_for_imshow(plot_matrix_10)
    # Compute symmetric colorbar limits around 0 (use 99th percentile for better contrast)
    data_absmax = np.nanpercentile(np.abs(plot_matrix_10), 99) if has_nans else np.percentile(np.abs(plot_matrix_10), 99)
    clim_10 = [-data_absmax, data_absmax] if data_absmax > 0 else [-1, 1]
    im10 = ax10.imshow(plot_matrix_10, vmin=clim_10[0], vmax=clim_10[1], cmap=redblue(), aspect='equal', interpolation='none', extent=_ext10)
    ax10.set_anchor('C')   # center the square matrix in its wide cell (below the
                           # objective panel) instead of letting it drift right.
    # Glue the colorbar to the right edge of the (square, aspect='equal') matrix.
    # A plain colorbar(ax=ax10) positions itself from ax10's full wide-cell bbox,
    # so it drifts to the right edge of the cell and collides with the next panel's
    # y-label. inset_axes uses ax10's axes-fraction coords, which track the drawn
    # square, keeping the colorbar snug against the matrix.
    cax10 = ax10.inset_axes([1.03, 0.0, 0.04, 1.0])
    plt.colorbar(im10, cax=cax10)
    ax10.set_title(title_10)
    ax10.set_xlabel('Units')
    ax10.set_ylabel('Units')

    # =========================================================================
    # Plot 11-12: Traces (use subsampled units and/or conditions if needed)
    # =========================================================================
    # Color conditions by mean response (handle NaNs)
    # Use subsampled conditions for coloring
    n_conds_to_plot = len(subsample_cond_idx)
    cond_means = np.nanmean(trial_avg, axis=0)
    cond_means_sub = cond_means[subsample_cond_idx]
    sorted_indices = np.argsort(cond_means_sub)
    colors = plt.cm.jet(np.linspace(0, 1, n_conds_to_plot))
    trace_colors = np.zeros((n_conds_to_plot, 3))
    for rank in range(n_conds_to_plot):
        cond_idx = sorted_indices[rank]
        trace_colors[cond_idx, :] = colors[rank, :3]  # Use only RGB, not alpha

    # Get subsampled data for traces (both units and conditions)
    trial_avg_sub = trial_avg[np.ix_(subsample_idx, subsample_cond_idx)]
    denoised_sub = denoised[np.ix_(subsample_idx, subsample_cond_idx)]

    # Trial-averaged traces
    ax11 = fig.add_subplot(gs[2, 2:4])  # Row 2, columns 2-3
    x_units = np.arange(len(subsample_idx))
    for c in range(n_conds_to_plot):
        ax11.plot(x_units, trial_avg_sub[:, c], color=trace_colors[c, :], linewidth=0.5)
    ax11.set_xlabel('Units')
    ax11.set_ylabel('Activity')
    ax11.set_title(f'Trial-Averaged Traces{subsample_suffix_traces}')
    ax11.grid(True)
    ax11.set_xlim([x_units[0], x_units[-1]])

    # Denoised traces
    ax12 = fig.add_subplot(gs[2, 4:6])  # Row 2, columns 4-5
    for c in range(n_conds_to_plot):
        ax12.plot(x_units, denoised_sub[:, c], color=trace_colors[c, :], linewidth=0.5)
    ax12.set_xlabel('Units')
    ax12.set_ylabel('Activity')
    ax12.set_title(f'PSN Denoised Traces{subsample_suffix_traces}')
    ax12.grid(True)
    ax12.set_xlim([x_units[0], x_units[-1]])

    # Match y-limits (handle NaNs) - use subsampled data for ylim
    all_trace_data = np.concatenate([trial_avg_sub.ravel(), denoised_sub.ravel()])
    y_min = np.nanmin(all_trace_data) if has_nans else np.min(all_trace_data)
    y_max = np.nanmax(all_trace_data) if has_nans else np.max(all_trace_data)
    y_range = y_max - y_min
    y_margin = y_range * 0.05

    ax11.set_ylim([y_min - y_margin, y_max + y_margin])
    ax12.set_ylim([y_min - y_margin, y_max + y_margin])

    # =========================================================================
    # Plot 13: Split-half reliability (use subsampling for scatter, full pop for means)
    # =========================================================================
    ax13 = fig.add_subplot(gs[2, 6:8])  # Row 2, columns 6-7

    # Split trials by odd/even indices (interleaved) to handle NaN patterns
    # where later trials may have more NaNs due to variable repetition counts
    odd_idx = np.arange(0, ntrials, 2)   # 0, 2, 4, ...
    even_idx = np.arange(1, ntrials, 2)  # 1, 3, 5, ...
    data_A = data[:, :, odd_idx]
    data_B = data[:, :, even_idx]

    # Trial averages (use nanmean to handle NaNs)
    tavg_A = np.nanmean(data_A, axis=2) if has_nans else np.mean(data_A, axis=2)
    tavg_B = np.nanmean(data_B, axis=2) if has_nans else np.mean(data_B, axis=2)

    # Denoise both splits
    unit_means = results['unit_means']

    # Apply denoiser via denoiser.T @ x (correct for all modes:
    # symmetric global denoisers have denoiser.T == denoiser,
    # and non-symmetric denoisers like hybrid/wiener need the transpose)
    dn_A = denoiser.T @ (tavg_A - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]
    dn_B = denoiser.T @ (tavg_B - unit_means[:, np.newaxis]) + unit_means[:, np.newaxis]

    # Compute per-unit split-half metric for ALL units
    metric_tavg = np.zeros(nunits)
    metric_cross = np.zeros(nunits)
    metric_dn = np.zeros(nunits)

    use_mse = (split_half_metric == 'mse')

    for u in range(nunits):
        if use_mse:
            # MSE: mean squared error across conditions
            mask = ~(np.isnan(tavg_A[u, :]) | np.isnan(tavg_B[u, :]))
            metric_tavg[u] = np.mean((tavg_A[u, mask] - tavg_B[u, mask])**2) if np.sum(mask) > 0 else np.nan

            mask_AB = ~(np.isnan(tavg_A[u, :]) | np.isnan(dn_B[u, :]))
            mask_BA = ~(np.isnan(dn_A[u, :]) | np.isnan(tavg_B[u, :]))
            mse_AB = np.mean((tavg_A[u, mask_AB] - dn_B[u, mask_AB])**2) if np.sum(mask_AB) > 0 else np.nan
            mse_BA = np.mean((dn_A[u, mask_BA] - tavg_B[u, mask_BA])**2) if np.sum(mask_BA) > 0 else np.nan
            metric_cross[u] = (mse_AB + mse_BA) / 2 if not (np.isnan(mse_AB) or np.isnan(mse_BA)) else np.nan

            mask_dn = ~(np.isnan(dn_A[u, :]) | np.isnan(dn_B[u, :]))
            metric_dn[u] = np.mean((dn_A[u, mask_dn] - dn_B[u, mask_dn])**2) if np.sum(mask_dn) > 0 else np.nan
        else:
            # Correlation (default)
            if np.nanstd(tavg_A[u, :]) > 0 and np.nanstd(tavg_B[u, :]) > 0:
                mask = ~(np.isnan(tavg_A[u, :]) | np.isnan(tavg_B[u, :]))
                if np.sum(mask) > 1:
                    metric_tavg[u] = np.corrcoef(tavg_A[u, mask], tavg_B[u, mask])[0, 1]
                else:
                    metric_tavg[u] = np.nan
            else:
                metric_tavg[u] = np.nan

            if (np.nanstd(tavg_A[u, :]) > 0 and np.nanstd(dn_B[u, :]) > 0 and
                np.nanstd(dn_A[u, :]) > 0 and np.nanstd(tavg_B[u, :]) > 0):
                mask_AB = ~(np.isnan(tavg_A[u, :]) | np.isnan(dn_B[u, :]))
                mask_BA = ~(np.isnan(dn_A[u, :]) | np.isnan(tavg_B[u, :]))
                if np.sum(mask_AB) > 1 and np.sum(mask_BA) > 1:
                    corr_AB = np.corrcoef(tavg_A[u, mask_AB], dn_B[u, mask_AB])[0, 1]
                    corr_BA = np.corrcoef(dn_A[u, mask_BA], tavg_B[u, mask_BA])[0, 1]
                    metric_cross[u] = (corr_AB + corr_BA) / 2
                else:
                    metric_cross[u] = np.nan
            else:
                metric_cross[u] = np.nan

            if np.nanstd(dn_A[u, :]) > 0 and np.nanstd(dn_B[u, :]) > 0:
                mask = ~(np.isnan(dn_A[u, :]) | np.isnan(dn_B[u, :]))
                if np.sum(mask) > 1:
                    metric_dn[u] = np.corrcoef(dn_A[u, mask], dn_B[u, mask])[0, 1]
                else:
                    metric_dn[u] = np.nan
            else:
                metric_dn[u] = np.nan

    # Subsampled metrics for plotting
    metric_tavg_sub = metric_tavg[subsample_idx]
    metric_cross_sub = metric_cross[subsample_idx]
    metric_dn_sub = metric_dn[subsample_idx]
    n_sub = len(subsample_idx)

    # Plot
    x_positions = np.array([1, 2, 3])
    labels = ['TAvg vs TAvg', 'TAvg vs Denoised', 'Denoised vs Denoised']

    # Add jitter for subsampled units
    x_jitter_sub = (np.random.rand(n_sub) - 0.5) * 0.16

    # Connecting lines (subsampled)
    for ii in range(n_sub):
        values = [metric_tavg_sub[ii], metric_cross_sub[ii], metric_dn_sub[ii]]
        if not np.any(np.isnan(values)):
            ax13.plot(x_positions + x_jitter_sub[ii], values,
                     color=[0.5, 0.5, 0.5], linewidth=0.3)

    # Scatter points (subsampled)
    ax13.scatter(x_positions[0] + x_jitter_sub, metric_tavg_sub, s=15, color='blue',
                alpha=0.4, zorder=2)
    ax13.scatter(x_positions[1] + x_jitter_sub, metric_cross_sub, s=15, color=[1, 0.84, 0],
                alpha=0.4, zorder=2)
    ax13.scatter(x_positions[2] + x_jitter_sub, metric_dn_sub, s=15, color=[0.5, 0.8, 0.3],
                alpha=0.4, zorder=2)

    # Medians (FULL population). Use the median (not the mean) so these central
    # markers match the recovery-tradeoff panel (fig7), which reports the median
    # per-unit split-half reliability for its trial-average / chosen / Wiener dots.
    med_tavg = np.nanmedian(metric_tavg)
    med_cross = np.nanmedian(metric_cross)
    med_dn = np.nanmedian(metric_dn)

    ax13.scatter(x_positions[0], med_tavg, s=100, color='blue',
                edgecolors='white', linewidths=2, zorder=3)
    ax13.scatter(x_positions[1], med_cross, s=100, color=[1, 0.84, 0],
                edgecolors='white', linewidths=2, zorder=3)
    ax13.scatter(x_positions[2], med_dn, s=100, color=[0.2, 0.6, 0.2],
                edgecolors='white', linewidths=2, zorder=3)

    # Labels (FULL population medians)
    all_vals_sub = np.concatenate([metric_tavg_sub, metric_cross_sub, metric_dn_sub])
    valid_vals = all_vals_sub[~np.isnan(all_vals_sub)]
    y_range_metric = (np.max(valid_vals) - np.min(valid_vals)) if len(valid_vals) > 0 else 1
    y_offset = y_range_metric * 0.06
    ax13.text(x_positions[0], med_tavg + y_offset, f'{med_tavg:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax13.text(x_positions[1], med_cross + y_offset, f'{med_cross:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax13.text(x_positions[2], med_dn + y_offset, f'{med_dn:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax13.set_xticks(x_positions)
    ax13.set_xticklabels(labels, rotation=0, fontsize=7)
    ax13.set_ylabel('MSE' if use_mse else 'Pearson r')

    metric_label = 'MSE' if use_mse else 'Reliability'
    if has_nans:
        valid_A = np.sum(~np.any(np.isnan(data_A), axis=0), axis=1)
        valid_B = np.sum(~np.any(np.isnan(data_B), axis=0), axis=1)
        avg_valid_A = np.mean(valid_A[valid_A > 0])
        avg_valid_B = np.mean(valid_B[valid_B > 0])
        ax13.set_title(f'Split-Half {metric_label}\n({avg_valid_A:.1f} vs {avg_valid_B:.1f} avg trials){subsample_suffix_units}')
    else:
        ax13.set_title(f'Split-Half {metric_label}\n({data_A.shape[2]} vs {data_B.shape[2]} trials){subsample_suffix_units}')

    ax13.grid(True)
    ax13.set_xlim([0.5, 3.5])
    if not use_mse:
        ax13.axhline(y=0, color='k', linewidth=1)

    # Set y-limits
    if len(valid_vals) > 0:
        y_min_c = np.min(valid_vals)
        y_max_c = np.max(valid_vals)
        y_range_c = y_max_c - y_min_c
        y_pad = max(0.1 if not use_mse else y_range_c * 0.1, y_range_c * 0.15)
        ax13.set_ylim([y_min_c - y_pad, y_max_c + y_pad])
    else:
        ax13.set_ylim([-1, 1] if not use_mse else [0, 1])

    # =========================================================================
    # Plot 14-17: Signal/Noise Diagnostics (use subsampling for scatter, full pop for means)
    # =========================================================================

    # Extract signal/noise variance data
    if 'svnv_before' in results and 'svnv_after' in results:
        sv_before = results['svnv_before'][:, 0]
        nv_before = results['svnv_before'][:, 1]
        sv_after = results['svnv_after'][:, 0]
        nv_after = results['svnv_after'][:, 1]

        # Compute noise-ceiling SNR (NCSNR). Rectify negative signal variance to 0.
        # Floor the denominator relative to the data (mirrors the per-dimension NCSNR
        # guard) so a unit whose noise variance underflows to ~0 cannot blow the ratio
        # up to ~1e7. Kept finite (no NaN) because the means/limits below consume these.
        nfloor_before = max(np.finfo(float).eps, 1e-8 * np.nanmax(nv_before))
        nfloor_after = max(np.finfo(float).eps, 1e-8 * np.nanmax(nv_after))
        ncsnr_before = np.sqrt(np.maximum(sv_before, 0)) / np.sqrt(np.maximum(nv_before, nfloor_before))
        ncsnr_after = np.sqrt(np.maximum(sv_after, 0)) / np.sqrt(np.maximum(nv_after, nfloor_after))

        # Compute noise ceiling percentage (use ntrials_avg for NaN data)
        noiseceiling_before = 100 * (ncsnr_before**2 / (ncsnr_before**2 + 1/ntrials_avg))
        noiseceiling_after = 100 * (ncsnr_after**2 / (ncsnr_after**2 + 1/ntrials_avg))

        # Subsampled versions for plotting
        sv_before_sub = sv_before[subsample_idx]
        sv_after_sub = sv_after[subsample_idx]
        nv_before_sub = nv_before[subsample_idx]
        nv_after_sub = nv_after[subsample_idx]
        ncsnr_before_sub = ncsnr_before[subsample_idx]
        ncsnr_after_sub = ncsnr_after[subsample_idx]
        noiseceiling_before_sub = noiseceiling_before[subsample_idx]
        noiseceiling_after_sub = noiseceiling_after[subsample_idx]
        n_sub = len(subsample_idx)

        # Define x positions
        x_before = 1
        x_after = 2
        x_jitter_diag = (np.random.rand(n_sub) - 0.5) * 0.1

        # Plot 14: Signal Variance
        ax14 = fig.add_subplot(gs[3, 0:2])  # Row 3, columns 0-1
        for ii in range(n_sub):
            ax14.plot([x_before, x_after] + x_jitter_diag[ii],
                     [sv_before_sub[ii], sv_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax14.scatter(x_before + x_jitter_diag, sv_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax14.scatter(x_after + x_jitter_diag, sv_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_sv_before = np.mean(sv_before)
        mean_sv_after = np.mean(sv_after)
        ax14.scatter(x_before, mean_sv_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax14.scatter(x_after, mean_sv_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically (use subsampled for range)
        y_range_sv = np.max([sv_before_sub.max(), sv_after_sub.max()]) - np.min([sv_before_sub.min(), sv_after_sub.min()])
        y_offset_sv = y_range_sv * 0.08

        ax14.text(x_before, mean_sv_before + y_offset_sv, f'{mean_sv_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax14.text(x_after, mean_sv_after + y_offset_sv, f'{mean_sv_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax14.set_xlim([0.5, 2.5])
        ax14.set_xticks([1, 2])
        ax14.set_xticklabels(['Before', 'After'])
        ax14.set_ylabel('Signal Variance')
        ax14.set_title(f'Signal Variance{subsample_suffix_units}')
        ax14.grid(True)

        # Plot 15: Noise Variance
        ax15 = fig.add_subplot(gs[3, 2:4])  # Row 3, columns 2-3
        for ii in range(n_sub):
            ax15.plot([x_before, x_after] + x_jitter_diag[ii],
                     [nv_before_sub[ii], nv_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax15.scatter(x_before + x_jitter_diag, nv_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax15.scatter(x_after + x_jitter_diag, nv_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_nv_before = np.mean(nv_before)
        mean_nv_after = np.mean(nv_after)
        ax15.scatter(x_before, mean_nv_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax15.scatter(x_after, mean_nv_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically
        y_range_nv = np.max([nv_before_sub.max(), nv_after_sub.max()]) - np.min([nv_before_sub.min(), nv_after_sub.min()])
        y_offset_nv = y_range_nv * 0.08

        ax15.text(x_before, mean_nv_before + y_offset_nv, f'{mean_nv_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax15.text(x_after, mean_nv_after + y_offset_nv, f'{mean_nv_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax15.set_xlim([0.5, 2.5])
        ax15.set_xticks([1, 2])
        ax15.set_xticklabels(['Before', 'After'])
        ax15.set_ylabel('Noise Variance / ntrials')
        ax15.set_title(f'Trial-Averaged Noise Variance{subsample_suffix_units}')
        ax15.grid(True)

        # Set unified ylims for both signal and noise variance plots (use subsampled for range)
        all_variance_vals_sub = np.concatenate([sv_before_sub, sv_after_sub, nv_before_sub, nv_after_sub])
        y_max_unified = np.max(all_variance_vals_sub)
        y_pad_unified = y_max_unified * 0.05  # 5% padding at bottom
        unified_ylim = [-y_pad_unified, y_max_unified + y_max_unified * 0.15]

        ax14.set_ylim(unified_ylim)
        ax14.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax15.set_ylim(unified_ylim)
        ax15.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        # Plot 16: NCSNR
        ax16 = fig.add_subplot(gs[3, 4:6])  # Row 3, columns 4-5
        for ii in range(n_sub):
            ax16.plot([x_before, x_after] + x_jitter_diag[ii],
                     [ncsnr_before_sub[ii], ncsnr_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax16.scatter(x_before + x_jitter_diag, ncsnr_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax16.scatter(x_after + x_jitter_diag, ncsnr_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_ncsnr_before = np.mean(ncsnr_before)
        mean_ncsnr_after = np.mean(ncsnr_after)
        ax16.scatter(x_before, mean_ncsnr_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax16.scatter(x_after, mean_ncsnr_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Calculate y_offset dynamically
        y_range_ncsnr = np.max([ncsnr_before_sub.max(), ncsnr_after_sub.max()]) - np.min([ncsnr_before_sub.min(), ncsnr_after_sub.min()])
        y_offset_ncsnr = y_range_ncsnr * 0.08

        ax16.text(x_before, mean_ncsnr_before + y_offset_ncsnr, f'{mean_ncsnr_before:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax16.text(x_after, mean_ncsnr_after + y_offset_ncsnr, f'{mean_ncsnr_after:.3f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Set ylims with padding (use subsampled for range)
        all_ncsnr_vals_sub = np.concatenate([ncsnr_before_sub, ncsnr_after_sub])
        y_max_ncsnr = np.max(all_ncsnr_vals_sub)
        y_pad_ncsnr_bottom = y_max_ncsnr * 0.05  # 5% padding at bottom
        y_pad_ncsnr_top = y_max_ncsnr * 0.15  # 15% padding at top
        ax16.set_ylim([-y_pad_ncsnr_bottom, y_max_ncsnr + y_pad_ncsnr_top])
        ax16.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax16.set_xlim([0.5, 2.5])
        ax16.set_xticks([1, 2])
        ax16.set_xticklabels(['Before', 'After'])
        ax16.set_ylabel('NCSNR')
        ax16.set_title(f'Noise Ceiling SNR (NCSNR){subsample_suffix_units}')
        ax16.grid(True)

        # Plot 17: Noise Ceiling %
        ax17 = fig.add_subplot(gs[3, 6:8])  # Row 3, columns 6-7
        for ii in range(n_sub):
            ax17.plot([x_before, x_after] + x_jitter_diag[ii],
                     [noiseceiling_before_sub[ii], noiseceiling_after_sub[ii]],
                     color=[0.7, 0.7, 0.7], linewidth=0.5)

        ax17.scatter(x_before + x_jitter_diag, noiseceiling_before_sub, s=40, color=[0.3, 0.5, 0.8],
                    alpha=0.6)
        ax17.scatter(x_after + x_jitter_diag, noiseceiling_after_sub, s=40, color=[0.8, 0.3, 0.3],
                    alpha=0.6)

        # Means from FULL population
        mean_nc_before = np.mean(noiseceiling_before)
        mean_nc_after = np.mean(noiseceiling_after)
        ax17.scatter(x_before, mean_nc_before, s=120, color=[0.1, 0.3, 0.6],
                    edgecolors='white', linewidths=2, zorder=3)
        ax17.scatter(x_after, mean_nc_after, s=120, color=[0.6, 0.1, 0.1],
                    edgecolors='white', linewidths=2, zorder=3)

        # Fixed y_offset for noise ceiling (percentage scale 0-100). Place the
        # label below the point when above it would collide with the title.
        y_offset_nc = 100 * 0.08

        def _nc_label(x, m):
            if m + y_offset_nc > 98:
                ax17.text(x, m - y_offset_nc, f'{m:.3f}', ha='center', va='top',
                          fontsize=10, fontweight='bold')
            else:
                ax17.text(x, m + y_offset_nc, f'{m:.3f}', ha='center', va='bottom',
                          fontsize=10, fontweight='bold')
        _nc_label(x_before, mean_nc_before)
        _nc_label(x_after, mean_nc_after)

        ax17.set_xlim([0.5, 2.5])
        ax17.set_ylim([-5, 100])  # Add negative padding to make yline at 0 visible
        ax17.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

        ax17.set_xticks([1, 2])
        ax17.set_xticklabels(['Before', 'After'])
        ax17.set_ylabel('Noise Ceiling (%)')

        if has_nans:
            ax17.set_title(f'Noise Ceiling Percentage ({ntrials_avg:.1f} avg trials){subsample_suffix_units}')
        else:
            ax17.set_title(f'Noise Ceiling Percentage ({ntrials} trials){subsample_suffix_units}')

        ax17.grid(True)

    # Save figure if figurepath specified, otherwise show it
    if figurepath is not None:
        fig.savefig(figurepath, dpi=150, bbox_inches='tight')
        # Don't show - just save and return (caller will close)
    else:
        plt.show()

    return fig
