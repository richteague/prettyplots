import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import griddata


def fill_NaN(data):
    """Fill in the NaN values with nearest values."""
    xgrid, ygrid = np.arange(data.shape[1]), np.arange(data.shape[0])
    xpnts, ypnts = np.meshgrid(xgrid, ygrid)
    xpnts, ypnts = xpnts.flatten(), ypnts.flatten()
    dpnts = data.flatten()
    mask = np.where(np.isfinite(dpnts))
    return griddata((xpnts[mask], ypnts[mask]), dpnts[mask],
                    (xgrid[None, :], ygrid[:, None]), method='nearest')


def gaussian(x, dx, A=1.0, x0=0.0, offset=0.0):
    """Gaussian function with standard deviation dx."""
    return A * np.exp(-0.5 * np.power((x-x0)/dx, 2.)) + offset


def gradient_between(x, y, dy, ax=None, **kwargs):
    """Similar to fill_between but with a gradient."""

    if ax is None:
        fig, ax = plt.subplots()

    # If dy has a shape (x.size, 2) or (2, x.size) then we assume that these
    # are percentiles and y is the median. If so, we can convert these to
    # uncertainties for the plotting. Otherwise we just double up the
    # uncertainties.

    try:
        ndim = dy.ndim
    except AttributeError:
        dy = np.squeeze(dy)
        ndim = dy.ndim

    if ndim == 1:
        dy = np.array([dy, dy])
    elif ndim == 2:
        if dy.shape[1] == 2:
            dy = dy.T
        elif dy.shape[0] != 2:
            raise ValueError("Wrong shaped uncertainties.")
        if kwargs.get('percentiles', False):
            dy = np.squeeze([y - dy[0], dy[1] - y])
    else:
        raise TypeError("'dy' must be only 1D or 2D.")
    if not (dy.shape[0] == 2 and dy.shape[1] == x.size):
        raise ValueError()

    # Populate the colours and styles of the plotting.
    # Colors, can set each individually but 'color' will override all.

    color = kwargs.pop('color', None)
    if color is not None:
        fc = color
        lc = color
        ec = [color]
    else:
        fc = kwargs.get('gradcolor', 'dodgerblue')
        lc = kwargs.get('linecolor', 'dodgerblue')
        ec = np.array([kwargs.get('edgecolor', 'k')]).flatten()

    # Controlls the gradient fill. The gradient will run from an alpha of
    # alphamax at the median to ~0 at y +/- nsigma * dy. This will be built
    # from nfill fill_between calls.

    am = kwargs.get('alphamax', .7)
    ns = kwargs.get('nsigma', 1)
    fn = kwargs.get('nfills', kwargs.get('nfill', 35))

    # Styles for the percentiles. Will cycle through the values if given in a
    # list. Should not complain if the lists are different lenghts.

    lw = kwargs.get('linewidth', 1.25)
    ms = kwargs.get('markersize', 3)
    ed = np.array([kwargs.get('edges', [1])]).flatten()
    ea = np.array([kwargs.get('edgealpha', [0.5, 0.25])]).flatten()
    es = np.array([kwargs.get('edgestyle', ':')]).flatten()

    # Incrementally calculate the alpha for a given layer and plot it.
    # Note that this doesn't work for logarithmic data thus far.

    alphacum = 0.0

    # TODO: Check whether this deals with logarithmic vales.
    if kwargs.get('log', False):
        fy = np.insert(np.logspace(-3, 0, fn) * ns, 0, 0)
    else:
        fy = np.linspace(0., ns, fn)

    for n in fy[::-1]:
        alpha = np.mean(gaussian(n, ns / 3., am)) - alphacum
        ax.fill_between(x, y-n*dy[0], y+n*dy[1], facecolor=fc, lw=0,
                        alpha=alpha)
        alphacum += alpha

    # Properties for the edges. These are able to be interated over.

    for e, edge in enumerate(ed):
        ax.plot(x, y-edge*dy[0], alpha=ea[e % len(ea)], color=ec[e % len(ec)],
                linestyle=es[e % len(es)], zorder=-1)
        ax.plot(x, y+edge*dy[1], alpha=ea[e % len(ea)], color=ec[e % len(ec)],
                linestyle=es[e % len(es)], zorder=-1)

    # Mean value including the label. Note that we do not call the legend here
    # so extra legend kwargs can be used if necessary. If 'outline' is true,
    # include a slightly thicker line in black.

    if kwargs.get('outline', True):
        ax.errorbar(x, y, color='k', fmt=kwargs.get('fmt', '-o'),
                    ms=ms, mew=1, lw=lw*2, zorder=1)
    ax.errorbar(x, y, color=lc, fmt=kwargs.get('fmt', '-o'),
                ms=ms, mew=0, lw=lw, label=kwargs.get('label', None), zorder=1)

    return ax


def gradient_fill(x, y, dy=None, region='below', ax=None, **kwargs):
    """Fill above or below a line with a gradiated fill."""
    if ax is None:
        fig, ax = plt.subplots()
    if dy is None:
        dy = y
    if region == 'below':
        ax = gradient_between(x, y, [dy, np.zeros(x.size)], ax=ax, **kwargs)
    elif region == 'above':
        ax = gradient_between(x, y, [np.zeros(x.size), dy], ax=ax, **kwargs)
    else:
        raise ValueError("Must set 'region' to 'above' or 'below'.")
    lc = kwargs.get('linecolor', 'k')
    ax.plot(x, y, color=lc)
    return ax


def powerlaw(x, x0, q, xc=1.0, dx0=0.0, dq=0.0):
    """Powerlaw including errors."""
    y = x0 * np.power(x / xc, q)
    if dx0 > 0.0 or dq > 0.0:
        dy = y * np.hypot(dx0 / x, dq * (np.log(x) - np.log(xc)))
        return y, dy
    return y


def texify(s, underscore='\ '):
    """Return a string which can be printed with TeX."""
    s = s.replace('_', underscore)
    ns = r'${\rm %s}$' % s
    return ns


def running_mean_convolve(y, ncells=2):
    """Returns the running mean over 'ncells' number of cells."""
    window = np.ones(ncells) / ncells
    return np.average([np.convolve(y[::-1], window, mode='same')[::-1],
                       np.convolve(y, window, mode='same')], axis=0)


def running_mean(y, window=5, x=None):
    """Calculate the running mean using a simple window."""

    # Define the window size.
    window = int(window)
    if window >= len(y):
        raise ValueError("Window too big.")

    # Sort the data if x values provided.
    if x is not None:
        x, y = sort_arrays(x, y)

    # Include dummy values.
    pad_low = y[0] * np.ones(window)
    pad_high = y[-1] * np.ones(window)
    y_pad = np.concatenate([pad_low, y, pad_high])

    # Define the window indices.
    a = int(np.ceil(window / 2))
    b = window - a

    # Loop through and calculate.
    mu = [np.nanmean(y_pad[i-a:i+b]) for i in range(window, len(y) + window)]
    return np.squeeze(mu)


def running_percentiles(y, percentiles=[16., 50., 84.], window=5, x=None):
    """Calculate the running percentile values within a window."""

    # Define the window size.
    window = int(window)
    if window >= len(y):
        raise ValueError("Window too big.")

    # Sort the data if x values provided.
    if x is not None:
        x, y = sort_arrays(x, y)

    # Include dummy values.
    pad_low = y[0] * np.ones(window)
    pad_high = y[-1] * np.ones(window)
    y_pad = np.concatenate([pad_low, y, pad_high])

    # Define the window indices.
    a = int(np.ceil(window / 2))
    b = window - a

    # Loop through and calculate.
    pcnt = [np.nanpercentile(y_pad[i-a:i+b], percentiles)
            for i in range(window, len(y) + window)]
    return np.squeeze(pcnt)


def running_stdev(y, window=5, x=None):
    """Calculate the running standard deviation within a window."""
    return running_variance(y, window, x)**0.5


def running_variance(y, window=5, x=None):
    """Calculate the running variance using a simple window."""

    # Define the window size.
    window = int(window)
    if window >= len(y):
        raise ValueError("Window too big.")

    # Sort the data if x values provided.
    if x is not None:
        x, y = sort_arrays(x, y)

    # Include dummy values.
    pad_low = y[0] * np.ones(window)
    pad_high = y[-1] * np.ones(window)
    y_pad = np.concatenate([pad_low, y, pad_high])

    # Define the window indices.
    a = int(np.ceil(window / 2))
    b = window - a

    # Loop through and calculate.
    var = [np.nanvar(y_pad[i-a:i+b]) for i in range(window, len(y) + window)]
    return np.squeeze(var)


def sort_arrays(x, y):
    """Sort the data for monotonically increasing x."""
    idx = np.argsort(x)
    return x[idx], y[idx]


def plotbeam(bmaj, bmin=None, bpa=0.0, ax=None, **kwargs):
    """Plot a beam. Input must be same units as axes. PA in degrees E of N."""
    if ax is None:
        fig, ax = plt.subplots()
    if bmin is None:
        bmin = bmaj
    if bmin > bmaj:
        temp = bmin
        bmin = bmaj
        bmaj = temp
    offset = kwargs.get('offset', 0.125)
    ax.add_patch(Ellipse(ax.transLimits.inverted().transform((offset, offset)),
                         width=bmin, height=bmaj, angle=-bpa,
                         fill=False, hatch=kwargs.get('hatch', '////////'),
                         lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                         color=kwargs.get('color', kwargs.get('c', 'k'))))
    return


def plotbeam_FWHM(beam, dx=0.9, dy=0.9, ax=None, **kwargs):
    """Plot the beam FWHM for a linear plot. Must be same units!"""
    if ax is None:
        fig, ax = plt.subplots()
    x, y = ax.transLimits.inverted().transform((dx, dy))
    ax.errorbar(x, y, xerr=0.5*beam, capsize=2.0, capthick=1.25,
                lw=kwargs.get('linewidth', kwargs.get('lw', 1.25)),
                color=kwargs.get('color', kwargs.get('c', 'k')))
    text = kwargs.get('text', False)
    if text:
        text = text if type(text) is not bool else '%.2f' % beam
        ax.text(x - 0.8 * beam, y, text, ha='right', va='center', fontsize=7)
    return


def percentiles_to_errors(pcnts):
    """Covert [16, 50, 84]th percentiles to [y, -dy, +dy]."""
    pcnts = np.squeeze([pcnts])
    if pcnts.ndim > 1:
        if pcnts.shape[1] != 3 and pcnts.shape[0] == 3:
            pcnts = pcnts.T
        if pcnts.shape[1] != 3:
            raise TypeError("Must provide a Nx3 or 3xN array.")
        return np.array([[p[1], p[1]-p[0], p[2]-p[1]] for p in pcnts]).T
    return np.squeeze([pcnts[1], pcnts[1]-pcnts[0], pcnts[2]-pcnts[1]])
