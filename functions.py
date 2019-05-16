import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.interpolate import griddata


rainbows = ['#e61b23', '#f8a20e', '#00a650', '#1671c2', '#282b80', '#000000']
mutedrwb = ['#1e1a31', '#6c2963', '#b54b76', '#e5697a', '#f98b74', '#ffb568',
            '#ffe293', '#ffffff', '#b8f7f7', '#70d3e4', '#5eadcd', '#548abe',
            '#4f63b1', '#4d4690', '#383353']


def step_PDF(x, x0, dx):
    """
    Step PDF function. Helps avoid the discontinuity in Gaussian PDFs.
    Args:
        x (ndarray[float]): Arrays of points to sample the PDF on.
        x0 (float): Median value of the PDF.
        dx (ndarray[float]): One-sigma limits calculated from the
            16th and 84th percentiles.
    Returns:
        PDF (ndarray[float]): Simple step function PDF from percentiles.
    """
    mask = np.logical_or(np.logical_and(x >= x0-3*dx[0], x <= x0-2*dx[0]),
                         np.logical_and(x >= x0+2*dx[1], x <= x0+3*dx[1]))
    PDF = np.where(mask, 2.1, 0.0)
    mask = np.logical_or(np.logical_and(x >= x0-2*dx[0], x <= x0-dx[0]),
                         np.logical_and(x >= x0+dx[1], x <= x0+2*dx[1]))
    PDF = np.where(mask, 13.6, PDF)
    mask = np.logical_and(x >= x0-dx[0], x <= x0+dx[1])
    PDF = np.where(mask, 64.0, PDF)
    return PDF / np.trapz(PDF, x)


def combine_samples(y, dy, Nx=10000):
    """
    Combine samples each with their own PDF distribution and
    the intrinsic scatter in their median values.
    Args:
        y (ndarray[float]): Medians of all the samples.
        dy (ndarray[float]): Array of the 1 sigma uncertainties
            for the samples based on their 16th to 84th percentiles.
            Must be either one for each y value, or two, with the
            first being the 16th to 50th percentile range and the
            latter being the 50th to 84th. If only a single value
            is given we assume symmetric values.
        Nx (optional[int]): Number of samples for the x axis. More
            samples result in a more accurate result.
    Returns:
        percentiles (ndarray[float]): The median and 1 sigma
            uncertainties based on the 16th to 84th percentile of
            the combined PDF.
    """

    # Check the shapes of the input arrays.
    dy = np.atleast_2d(dy)
    if dy.shape[1] != y.shape[0]:
        if dy.shape[0] == y.shape[0]:
            dy = dy.T
        else:
            raise ValueError("Wrong uncertainty shape.")
    if dy.shape[0] == 1:
        dy = np.vstack([dy, dy])
    elif dy.shape[0] > 2:
        raise ValueError("Too many uncertainties per sample.")
    dy = abs(dy).T

    # Make the x-axis.
    x = abs(y).max() + 4.0 * dy.max()
    x = np.linspace(-x, x, int(Nx)) + np.nanmean(y)

    # Fill with all the PDFs and calculate percentiles.
    yc = [step_PDF(x, x0, dx) for x0, dx in zip(y, dy)]
    yc = np.cumsum(np.sum(yc, axis=0))
    yc /= yc[-1]
    p = [x[abs(yc - p).argmin()] for p in [0.16, 0.5, 0.84]]
    return np.array([p[1], p[1] - p[0], p[2] - p[1]])


def read_samples(fn, collapse=0, pdf_axis=-1, percentiles=True):
    """
    Read in the data and combine the PDFs.

    Args:
        fn (path): Path to the .npy file of samples.
        collapse (int): Axis along which to combine.
        pdf_axis (int): Axis containing the PDF.
        percentiles (bool): Does the percentiles_axis contain
            the [16, 50, 84] percentiles of the PDF? If not,
            assume [y, -dy, +dy].

    Returns:
        data (ndarray[float]): Array of combined PDFs.
    """
    data = np.load(fn)
    data = np.moveaxis(data, [collapse, pdf_axis], [0, 1])
    data_shape = data.shape
    data = data.reshape(*data.shape[:2], -1).T
    if percentiles:
        data = [combine_samples(d[1], [d[1] - d[0], d[2] - d[1]]) for d in data]
    else:
        data = [combine_samples(d[0], d[1:]) for d in data]
    data = np.array(data).T
    data = data.reshape(data_shape[1:])
    data = np.moveaxis(data, 0, pdf_axis)
    return data


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


def plotscale(scale, dx=0.9, dy=0.9, ax=None, text=None, text_above=True,
              **kwargs):
    """Plot a linear scale on the provided axes."""

    # Generate axes if not provided.
    if ax is None:
        fig, ax = plt.subplots()

    # Draw the scale bar.
    x, y = ax.transLimits.inverted().transform((dx, dy))
    ax.errorbar(x, y, xerr=0.5*scale, capsize=1.5, capthick=1.0,
                lw=kwargs.get('linewidth', kwargs.get('lw', 1.0)),
                color=kwargs.get('color', kwargs.get('c', 'k')))

    # Include the labelling.
    if text:
        if text_above:
            x, y = ax.transLimits.inverted().transform((dx, 1.2 * dy))
        else:
            x, y = ax.transLimits.inverted().transform((dx, 0.8 * dy))
        text = text if type(text) is not bool else '%.2f' % scale
        ax.text(x, y, text, ha='center', va='bottom' if text_above else 'top',
                fontsize=kwargs.get('fontsize', kwargs.get('fs', 7.0)),
                color=kwargs.get('color', kwargs.get('c', 'k')))
    return


def clip_array(arr, min=None, max=None, NaN=True):
    """Return a clipped array."""
    if NaN:
        arr = fill_NaN(arr)
    NaN_mask = np.isnan(arr)
    if min is not None:
        arr = np.where(arr >= min, arr, min)
        arr = np.where(NaN_mask, np.nan, arr)
    if max is not None:
        arr = np.where(arr <= max, arr, max)
        arr = np.where(NaN_mask, np.nan, arr)
    return arr


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


def step_PDF(x, x0, dx):
    """
    Step PDF function. Helps avoid the discontinuity in Gaussian PDFs.

    Args:
        x (ndarray[float]): Arrays of points to sample the PDF on.
        x0 (float): Median value of the PDF.
        dx (ndarray[float]): One-sigma limits calculated from the
            16th and 84th percentiles.

    Returns:
        PDF (ndarray[float]): Simple step function PDF from percentiles.
    """
    mask = np.logical_or(np.logical_and(x >= x0-3*dx[0], x <= x0-2*dx[0]),
                         np.logical_and(x >= x0+2*dx[1], x <= x0+3*dx[1]))
    PDF = np.where(mask, 2.1, 0.0)
    mask = np.logical_or(np.logical_and(x >= x0-2*dx[0], x <= x0-dx[0]),
                         np.logical_and(x >= x0+dx[1], x <= x0+2*dx[1]))
    PDF = np.where(mask, 13.6, PDF)
    mask = np.logical_and(x >= x0-dx[0], x <= x0+dx[1])
    PDF = np.where(mask, 64.0, PDF)
    return PDF / np.trapz(PDF, x)


def combine_samples(y, dy):
    """
    Combine samples each with their own PDF distribution and
    the intrinsic scatter in their median values.

    Args:
        y (ndarray[float]): Medians of all the samples.
        dy (ndarray[float]): Array of the 1 sigma uncertainties
            for the samples based on their 16th to 84th percentiles.
            Must be either one for each y value, or two, with the
            first being the 16th to 50th percentile range and the
            latter being hte 50th to 84th. If only a single value
            is given we assume symmetric values.

    Returns:
        percentiles (ndarray[float]): The median and 1 sigma
            uncertainties based on the 16th to 84th percentile of
            the combined PDF.
    """

    # Check the shapes of the input arrays.
    dy = np.atleast_2d(dy)
    if dy.shape[1] != y.shape[0]:
        if dy.shape[0] == y.shape[0]:
            dy = dy.T
        else:
            raise ValueError("Wrong uncertainty shape.")
    if dy.shape[0] == 1:
        dy = np.vstack([dy, dy])
    elif dy.shape[0] > 2:
        raise ValueError("Too many uncertainties per sample.")
    dy = abs(dy).T

    # Make the x-axis.
    x = 1.2 * (abs(y).max() + abs(dy).max())
    x = np.linspace(-x, x, 1000) + np.nanmean(y)

    # Fill with all the PDFs and calculate percentiles.
    yc = [step_PDF(x, x0, dx) for x0, dx in zip(y, dy)]
    yc = np.cumsum(np.sum(yc, axis=0))
    yc /= yc[-1]
    p = [x[abs(yc - p).argmin()] for p in [0.16, 0.5, 0.84]]
    return np.array([p[1], p[1] - p[0], p[2] - p[1]])


def superscript(name):
    """Return a LaTeX string with all numbers superscript."""
    s = [r'{\rm %s}' % c if not c.isdigit() else r'^{%s}' % c for c in name]
    return r'$%s$' % (''.join(s))


def subscript(name):
    """Return a LaTeX string with all the numbers subscript."""
    s = [r'{\rm %s}' % c if not c.isdigit() else r'_{%s}' % c for c in name]
    return r'$%s$' % (''.join(s))
