"""
Functions to help generate simple Gaussian Process fits to noisy data.
"""

import celerite
import numpy as np
from scipy.optimize import minimize


def SHO_model(x, y, dy, oversample=True, noise=True, return_var=False):
    """
    Return a model using a stochastically driven, dampled harmonic oscillator
    kernel. Using high Q values increases the oscillatory nature of the kernel,
    that is, points will become anti-correlated at specific spacings. This is
    therefore not a very good option unless you think this describes the data.

    - Input -

    x:          Sampling locations.
    y:          Observations.
    dy:         Uncertainties.
    oversample: If true, sample the GP at a higher resolution.
    noise:      Include a Jitter term in the GP model.
    return_var: If true, return the variance in the model too.

    - Returns -

    xx:         Sampling points of the GP
    yy:         Samples of the GP. Will also include the variance if return_var
                if True.
    """

    # Make sure arrays are ordered.
    x, y, dy = check_ordered(x, y, dy)

    # Define the Kernel.
    lnQ = np.log(1. / np.sqrt(2.))
    lnS0 = np.log(np.nanvar(y))
    lnw0 = np.log(np.nanmean(abs(np.diff(x))))

    kernel = celerite.terms.SHOTerm(log_S0=lnS0, log_Q=lnQ, log_omega0=lnw0)

    if noise:
        if np.nanmean(dy) != 0.0:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanmean(dy)))
        else:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanstd(y)))
        kernel = kernel + noise
    gp = celerite.GP(kernel, mean=np.nanmean(y), fit_mean=False)
    gp.compute(x, dy)

    # Minimize the results.
    params = gp.get_parameter_vector()
    params += 1e-3 * np.random.randn(len(params))
    soln = minimize(neg_log_like, params, jac=grad_neg_log_like,
                    args=(y, gp), method='L-BFGS-B')
    gp.set_parameter_vector(soln.x)

    # Define the new sampling rate for the GP.
    if oversample:
        if type(oversample) is float or type(oversample) is int:
            xx = np.linspace(x[0], x[-1], oversample * x.size)
        else:
            xx = np.linspace(x[0], x[-1], 5. * x.size)
    else:
        xx = x

    if soln.success:
        print('Solution:', soln.x)
        yy = gp.predict(y, xx, return_cov=False, return_var=return_var)
        if return_var:
            return xx, yy[0], yy[1]**0.5
        return xx, yy
    else:
        print('No Solution.')
        if return_var:
            return x, y, np.zeros(x.size)
        return x, y


def Matern32_model(x, y, dy, fit_mean=True, jitter=True, oversample=True,
                   return_var=True, verbose=False):
    """
    Return a model using a Matern 3/2 kernel. Most simple model possible.

    - Input -

    x:          Sampling locations.
    y:          Observations.
    dy:         Uncertainties.
    fit_mean:   Fit the mean of the observations.
    jitter:     Include a Jitter term in the GP model.
    oversample: If true, sample the GP at a higher resolution. If true will use
                a default of 5, otherwise a value can be specified.
    return_var: Return the variance of the fit.
    verbose:    Print out the best-fit values.

    - Returns -

    xx:         Sampling points of the GP
    yy:         Samples of the GP.
    zz:         Variance of the GP fit if return_var is True.
    """

    # Make sure arrays are ordered.
    x, y, dy = check_ordered(x, y, dy)

    # Define the Kernel.
    lnsigma, lnrho = np.log(np.nanstd(y)), np.log(np.nanmean(abs(np.diff(x))))
    kernel = celerite.terms.Matern32Term(log_sigma=lnsigma, log_rho=lnrho)

    if jitter:
        if np.nanmean(dy) != 0.0:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanmean(dy)))
        else:
            noise = celerite.terms.JitterTerm(log_sigma=np.log(np.nanstd(y)))
        kernel = kernel + noise
    gp = celerite.GP(kernel, mean=np.nanmean(y), fit_mean=fit_mean)
    gp.compute(x, dy)

    # Minimize the results.
    params = gp.get_parameter_vector()
    params += 1e-2 * np.random.randn(len(params))
    soln = minimize(neg_log_like, params, jac=grad_neg_log_like,
                    args=(y, gp), method='L-BFGS-B')
    gp.set_parameter_vector(soln.x)

    # Define the new sampling rate for the GP.
    if oversample:
        if type(oversample) is float or type(oversample) is int:
            xx = np.linspace(x[0], x[-1], oversample * x.size)
        else:
            xx = np.linspace(x[0], x[-1], 5. * x.size)
    else:
        xx = x

    if soln.success:
        if verbose:
            print('Solution:', soln.x)
        yy = gp.predict(y, xx, return_cov=False, return_var=return_var)
        if return_var:
            return xx, yy[0], yy[1]**0.5
        return xx, yy
    else:
        if verbose:
            print('No Solution.')
        if return_var:
            return x, y, np.zeros(x.size)
        return x, y


def check_ordered(x, y, dy):
    """Return ordered arrays."""
    idxs = np.argsort(x)
    return x[idxs], y[idxs], dy[idxs]


def neg_log_like(params, y, gp):
    """Negative log-likelihood fucntion."""
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)


def grad_neg_log_like(params, y, gp):
    """Gradient of the negative log-likelihood function."""
    gp.set_parameter_vector(params)
    return -gp.grad_log_likelihood(y)[1]
