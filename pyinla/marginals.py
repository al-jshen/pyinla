import numpy as np
import pandas as pd
import rpy2.robjects as ro

from pyinla.convert import convert_r2py
from pyinla.utils import rinla


def marginal_summary(marginal):
    """
    Return a summary of the marginal distribution.
    """
    return pd.DataFrame(convert_r2py(rinla.inla_zmarginal(marginal, silent=True)))


def marginal_logpdf(marginal, x):
    """
    Calculate the log PDF of the marginal distribution at locations x.
    """
    return rinla.inla_dmarginal(x, marginal, log=True)


def marginal_pdf(marginal, x):
    """
    Calculate the PDF of the marginal distribution at locations x.
    """
    return rinla.inla_dmarginal(x, marginal, log=False)


def marginal_cdf(marginal, x):
    """
    Calculate the CDF of the marginal distribution at locations x.
    """
    return rinla.inla_pmarginal(x, marginal)


def marginal_quantile(marginal, q):
    """
    Compute the quantile of the marginal distribution at quantiles q.
    """
    return rinla.inla_qmarginal(q, marginal)


def marginal_spline(marginal):
    """
    Return a spline representation of the marginal distribution.
    """
    return np.asarray(rinla.inla_smarginal(marginal))


def marginal_sample(marginal, n):
    """
    Draw n samples from the marginal distribution.
    """
    return rinla.inla_rmarginal(n, marginal)


def marginal_mode(marginal):
    """
    Return the mode of the marginal distribution.
    """
    return rinla.inla_mmarginal(marginal)


def marginal_expectation(marginal, fn):
    """Calculate the expectation of the marginal distribution with respect to
    the function.

    The function should be a string that can be evaluated in R.
    """
    return rinla.inla_emarginal(ro.r("function(x)" + fn), marginal)


def marginal_transform(marginal, fn):
    """Transform the marginal distribution with the function.

    The function should be a string that can be evaluated in R.
    """
    return rinla.inla_tmarginal(ro.r("function(x)" + fn), marginal)


def marginal_ci(marginal, ci):
    """
    Return the values between which the confidence interval is contained.
    """
    return rinla.inla_hpdmarginal(ci, marginal).reshape(2)
