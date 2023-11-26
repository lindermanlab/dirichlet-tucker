"""Killfish data specific visualization functions."""

from collections import OrderedDict
import itertools
import numpy as onp
import pandas as pd

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mplc
from scipy.cluster.hierarchy import linkage, leaves_list    # For hierarchically clustering topics

class ScalarMappable(mpl.cm.ScalarMappable):
    """Add callable functionality to mplc.ScalarMappable mixin-class."""
    
    def __init__(self, norm=None, cmap=None):
        """
        Parameters
        ----------
        norm : `.Normalize` (or subclass thereof) or str or None
            The normalizing object which scales data, typically into the
            interval ``[0, 1]``.
            If a `str`, a `.Normalize` subclass is dynamically generated based
            on the scale with the corresponding name.
            If *None*, *norm* defaults to a *colors.Normalize* object which
            initializes its scaling based on the first data processed.
        cmap : str or `~matplotlib.colors.Colormap`
            The colormap used to map normalized data values to RGBA colors.
        """
        super().__init__(norm=norm, cmap=cmap)

    def __call__(self, x):
        return self.to_rgba(x)


def multicolored_lineplot(x, y, c, cmap: mplc.Colormap, norm: mplc.Normalize, alpha: float=1):
    """Create LineCollection colored by the value of `c` in the normalized colormap space.

    The `vmin` and `vmax` parameters of `norm` must be set to correspond with
    expected values of `c` for proper color mapping.
    
    Add to plot via `ax.add_collection(lc)`

    Parameters
        x: ndarray, shape (N,)
        y: ndarray, shape (N,)
        c: ndarray, shape (N,)
        cmap: mplc.Colormap
        norm: mplc.Normalize
        alpha: float=1

    Reference:
    https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
    """
    
    # Array of x-y points, shape (N,1,2)
    xy_pts = onp.array([x, y]).T.reshape(-1,1,2)

    # Array of x-y line segments, shape (N-1, 1, 2)
    segments = onp.concatenate([xy_pts[1:], xy_pts[:-1]], axis=1)

    # Creae a LineCollection that uses the specified cmap
    lc = mpl.collections.LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(c[1:])
    lc.set_alpha(alpha)

    return lc
    
# ==============================================================================
# Syllable permutations, based on KL-divergence of syllable parameters
# - SYLLABLE_PERM_DICT: Consists of (cluster_name: indices) items and is used
#                       for annotating figures (see `set_syllable_cluster_ticks`)
# - SYLLABLE_PERM: Used for permutating syllable parameters
SYLLABLE_PERM_DICT = OrderedDict([
    ('inactive', [91, 97, 30, 10, 1, 43, 69] + [33,] + [99,]),                   # 99: Belly
    ('pause and drift', [32, 51, 82, 0, 56, 14, 19, 84, 92, 7, 66, 3, 17, 57, 85, 46, 27, 65]),
    ('straight swim', [9, 31, 28, 80, 68, 73, 6, 54, 58, 70, 39, 40, 25, 42, 89, 45, 77, 24, 60, 74, 94]),
    ('edge+sidebody', [63, 72, 15, 98, 23, 96, 36, 83, 55, 90, 76, 61, 88] + [50,]),
    ('j-turn+reverse', [44, 75, 59, 95, 48, 78, 18, 16, 35, 47],),
    ('aggression+glass surf', [71, 11, 64, 81, 87, 20, 21, 37, 8, 41, 52, 26, 93, 38, 12, 22, 5]),
    ('nose down', [86, 49, 67, 79, 2, 13, 62, 29, 34, 4, 53]),
])
SYLLABLE_PERM = list(itertools.chain.from_iterable(SYLLABLE_PERM_DICT.values()))

def set_syllable_cluster_ticks(ax=None, axis='x', font_kws={'fontsize': 'small'}):
    """Label specified axis with syllable _cluster_ names."""

    if ax is None:
        ax = plt.gca()

    cluster_names = list(SYLLABLE_PERM_DICT.keys())
    cluster_sizes = [len(v) for v in SYLLABLE_PERM_DICT.values()]
    
    maj_ticks = onp.cumsum([0,] + cluster_sizes) - 0.5
    min_ticks = onp.diff(maj_ticks) / 2 + maj_ticks[:-1]

    if axis == 'x':
        # Draw major ticks
        ax.set_xticks(maj_ticks)
        ax.tick_params(axis='x', which='major', length=5, labelbottom=False,)
        ax.grid(visible=True, which='major', axis='x', alpha=0.2, lw=0.5)

        # Annotate between the ticks
        ax.set_xticks(min_ticks, cluster_names, minor=True, **font_kws)
        ax.tick_params(axis='x', which='minor', bottom=False, labelbottom=True)
    else:
        # Draw major ticks
        ax.set_yticks(maj_ticks)
        ax.tick_params(axis='y', which='major', length=5, labelleft=False,)
        ax.grid(visible=True, axis='y', which='major', alpha=0.2, lw=0.5)

        # Annotate between the ticks
        ax.set_yticks(min_ticks, cluster_names, minor=True, **font_kws)
        ax.tick_params(axis='y', which='minor', left=False, labelleft=True)
    
    return ax

def draw_syllable_factors(params, autosort=True, ax=None):
    # Permute syllables to match our KL-clustering for better interpretability
    syllable_factors = params[3][:,SYLLABLE_PERM]
    K, D = syllable_factors.shape

    # Use hiearchical clustering on syllable factor ("behavioral topic") axis
    if autosort:
        method = 'centroid'
        metric = 'euclidean'
        topic_perm = leaves_list(linkage(syllable_factors, method, metric)).astype(int)

        syllable_factors = syllable_factors[topic_perm,:]
    else:
        topic_perm = onp.arange(K)

    # ------------------------------------------------------------------------
    ax = plt.gca() if ax is None else ax

    im = ax.imshow(syllable_factors, interpolation='none', aspect='auto',
                   cmap='magma', norm=mplc.LogNorm(0.5/D, 1.0))
    set_syllable_cluster_ticks(ax)
    plt.colorbar(im, ax=ax, extend='min')

    # Visually demarcate each behavioral topic
    ax.set_yticks(onp.arange(K)-0.5, [])
    ax.grid(visible=True, which='major', axis='y', alpha=0.8, lw=0.5)
    ax.set_ylabel('syllable factors / "behavioral topics"')

    return topic_perm

# ==============================================================================
def make_tod_series(freq):
    """Make time-of-day datetime.time points spaced at the given frequency.

    If `resample_rule='30min'`, then constructs returns array consisting of
        [(00:00:00), (00:30:00), ..., (23:00:00), (23:30:00)]

    Parameters
        freq: Pandas timedelta offset or offset string alias. See:
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """

    last_label = (
        (pd.to_datetime("00:00:00") - pd.Timedelta(freq)).to_pydatetime()
                                                         .time()
                                                         .strftime(format='%H:%M:%S')
    )

    return pd.date_range('00:00:00', last_label, freq=freq).time

def draw_circadian_bases(params, tod_freq='2H', autosort=True, axs=None):
    circadian_bases = params[2]
    D, K = circadian_bases.shape

    # Permute the circadian bases so that they are sorted by earliest peak
    if autosort:
        t_peak = onp.argmax(circadian_bases, axis=0)
        basis_perm = onp.argsort(t_peak, kind='stable')
        circadian_bases = circadian_bases[:, basis_perm]

    # Share a common a y-axis
    ymax = circadian_bases.max()

    # ------------------------------------------------------------------------
    if axs is None:
        _, axs = plt.subplots(nrows=K, ncols=1, squeeze=True,
                              gridspec_kw={'hspace':0.1}, figsize=(8,9), dpi=96)
    assert len(axs) == K, f'Expected {len(axs)} axes, expected {K}, i.e. one per basis'
        
    for k, ax in enumerate(axs):
        # Plot basis, and adjust x-axis days with human-interpretable times
        ax.plot(circadian_bases[:,k])
    
        # Grey out background if factor L2 norm is below a threshold
        mag = onp.linalg.norm(circadian_bases[:,k])
        if mag <= 0.3:
            ax.set_facecolor('0.8')
            ax.annotate(f'|factor|={mag:.2f}', (0.01,0.9), xycoords='axes fraction',
                        va='top', fontsize='small')
        
        # Label x-axis with time-of-day from 0H - 24H, every 2H
        t_dts = make_tod_series(tod_freq)
        t_locs = onp.concatenate([onp.linspace(0, D, num=len(t_dts), endpoint=False), [D]])
        t_labels = list(map(lambda dt: dt.strftime('%H'), t_dts)) + ['24']
        
        ax.set_xticks(t_locs)
        ax.set_xticklabels(t_labels)

        # Label y-axis with "circadian bases"
        if k == K // 2:
            ax.set_ylabel('time-of-day factors / "circadian bases"')

        # Set axis limits; reduce blank space margins
        ax.set_ylim(bottom=-0.1, top=1.1*ymax)
        ax.margins(x=0.01, y=0.5)

        # Draw time-of-day ticks; only annotate bottom-most subplot
        ax.tick_params(labelleft=False, labelbottom=False)
        if k == K-1:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel('time of day [24hr]')
        
        # Despine
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    return plt.gcf()

# ==============================================================================
# Blue to green to yellow
LIFESPAN_PALETTE = sns.color_palette("blend:#5A99AD,#7FAB5E,#C7B069", as_cmap=True)
LIFESPAN_PALETTE.set_under(color='#5A99AD')

# PowerNorm with gamma > 1 spreads large number out, providing more resolution
# Clip ages outside of 0.10-0.95 quantile
ABS_AGE_NORM = mplc.PowerNorm(gamma=1.5, vmin=105, vmax=340, clip=True)

# This colormap is both mappable (i.e. for colorbars) and callable (for cmap)
LIFESPAN_CMAP = ScalarMappable(cmap=LIFESPAN_PALETTE, norm=ABS_AGE_NORM)

def make_lifespan_colobar(**kwargs):
    """Function to call fixed version of colorbar.

    Issue
    -----
    plt.colorbar(LIFESPAN_CMAP, extend='both') results in the lower extension
    being yellow, which is the `over` limit, instead of the blue `under` limit.
    
    This is because in L1082 of matplotlib.colorbar,
    This is because the matplotlib colorbar boundary is extended by subtracting
    1 from the current boundary (of 0), then pass
    through the inverse of norm. However, this results in an (negative) overflow
    which leads to the bottom triangle being 'yellow'.
    To resolve this, we manually compute the boundaries, and associated values.

    https://github.com/matplotlib/matplotlib/blob/eb02b108ea181930ab37717c75e07ba792e01f1d/lib/matplotlib/colorbar.py#L1082C26-L1082C26
    """

    # Generate default boundaries, port of `mpl.colobar._uniform_y
    b = onp.linspace(0, 1, LIFESPAN_CMAP.cmap.N+1)

    # Add extra boundaries because we are extending both ends
    # But make sure to not subtract 1 from b[0], as original code does
    b = onp.hstack([b[0], b, b[-1]+1])

    # Transform form [0,1] back to cmap range
    b = LIFESPAN_CMAP.norm.inverse(b)

    # Calculate values between boundaries
    v = 0.5 * (b[:-1] + b[1:])

    # Make colorbar
    bad_kws = ['extend', 'boundaries', 'values', 'ticks', 'format']
    cbar = plt.colorbar(LIFESPAN_CMAP,
                        extend='both', boundaries=b, values=v,
                        ticks=onp.arange(120, 360, 30), format=mpl.ticker.ScalarFormatter(),
                        **{k: v for k, v in kwargs.items() if k not in bad_kws})
    
    return cbar
