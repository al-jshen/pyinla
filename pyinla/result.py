import functools

import matplotlib.pyplot as plt
import numpy as np
from rpy2.robjects.vectors import ListVector

from pyinla.convert import R_NULL, convert_r2py
from pyinla.marginals import *
from pyinla.utils import inla_summary


class MarginalType:
    def __init__(self, marginal_type):
        self.marginal_type = marginal_type
        self.names = list(self.marginal_type.names)

    def list_marginals(self):
        return self.names

    def get_marginal(self, key: str | int):
        if isinstance(key, int):
            return Marginals(self.marginal_type.rx2(self.names[key]))
        elif isinstance(key, str):
            return Marginals(self.marginal_type.rx2(key))
        else:
            raise TypeError("key must be either int or str")

    def apply(self, func, *args, **kwargs):
        results = []
        for m in range(len(self.names)):
            results.append(func(self.get_marginal(m), *args, **kwargs))
        return results


def check_2d(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if np.ndim(args[0]) != 2:
            assert np.ndim(args[0]) == 3, "Only 2D or 3D arrays are supported."
            raise ValueError(
                "Marginal must be 2d. You have a 3d marginal, which you can index into to get a 2d marginal."
            )
        return f(*args, **kwargs)

    return wrapper


class Marginals(np.ndarray):
    def __new__(cls, a):
        obj = np.asarray(a).view(cls)
        if isinstance(a, ListVector):
            cls.names = list(a.names)
        return obj

    def to_dict(self):
        return {k: v for k, v in zip(self.names, self)}

    @check_2d
    def summary(self):
        """
        Return a summary of the marginal distribution.
        """
        return marginal_summary(self)

    @check_2d
    def mode(self):
        """
        Return the mode of the marginal distribution.
        """
        return marginal_mode(self)

    @check_2d
    def logpdf(self, x):
        """
        Calculate the log PDF of the marginal distribution at locations x.
        """
        return marginal_logpdf(self, x)

    @check_2d
    def pdf(self, x):
        """
        Calculate the PDF of the marginal distribution at locations x.
        """
        return marginal_pdf(self, x)

    @check_2d
    def cdf(self, x):
        """
        Calculate the CDF of the marginal distribution at locations x.
        """
        return marginal_cdf(self, x)

    @check_2d
    def quantile(self, q):
        """
        Compute the quantile of the marginal distribution at quantiles q.
        """
        return marginal_quantile(self, q)

    @check_2d
    def ci(self, ci):
        """
        Compute the values within which the confidence interval is contained.
        """
        return marginal_ci(self, ci)

    @check_2d
    def sample(self, n):
        """
        Return a spline representation of the marginal distribution.
        """
        return marginal_sample(self, n)

    @check_2d
    def expectation(self, fn):
        """
        Calculate the expectation of the marginal distribution with respect to the function.
        The function should be a string that can be evaluated in R.
        """
        return marginal_expectation(self, fn)

    @check_2d
    def transform(self, fn):
        """
        Transform the marginal distribution with the function.
        The function should be a string that can be evaluated in R.
        """
        return Marginals(marginal_transform(self, fn))

    @check_2d
    def spline(self):
        """
        Return an interpolated spline representation of the marginal distribution.
        """
        return Marginals(marginal_spline(self).T)

    @check_2d
    def plot(self, **kwargs):
        """
        Plot the marginal distribution.
        """
        plt.plot(self[:, 0], self[:, 1], **kwargs)


class Result:
    def __init__(self, result):
        self.result = result
        self.suffixes = [
            "fixed",
            "lincomb",
            "lincomb.derived",
            "random",
            "linear.predictor",
            "fitted.values",
            "hyperpar",
            "spde2.blc",
            "spde3.blc",
        ]
        self.names = list(self.result.names)

    def __repr__(self):
        return str(inla_summary(self.result))

    def __getitem__(self, key):
        return self.result.rx2(key)

    def get(self, key):
        return self.result.rx2(key)

    def get_py(self, key):
        return convert_r2py(self.result.rx2(key))

    def get_marginal_type(self, kind):
        assert kind in self.suffixes, "kind must be one of {}".format(self.suffixes)
        mt = self.result.rx2("marginals." + kind)
        if mt == R_NULL:
            return None
        return MarginalType(self.get("marginals." + kind))

    def get_summary(self, kind):
        assert kind in self.suffixes, "kind must be one of {}".format(self.suffixes)
        st = self.result.rx2("summary." + kind)
        if st == R_NULL:
            return None
        return convert_r2py(self.get("summary." + kind))

    def list_marginal_types(self):
        return self.suffixes

    def list_summaries(self):
        return self.suffixes

    def list_marginals(self, kind):
        return convert_r2py(self.get_marginal_type(kind).names)

    def sample_posterior(self, n: int):
        ps = dict(
            hyperpar=[],
            latent=[],
            logdens_hyperpar=[],
            logdens_latent=[],
            logdens_joint=[],
        )
        post = convert_r2py(rinla.inla_posterior_sample(n, self.result))

        for p in post:
            ps["hyperpar"].append(p["hyperpar"])
            ps["latent"].append(p["latent"])
            ps["logdens_hyperpar"].append(p["logdens"]["hyperpar"])
            ps["logdens_latent"].append(p["logdens"]["latent"])
            ps["logdens_joint"].append(p["logdens"]["joint"])

        for k in ps:
            ps[k] = np.stack(ps[k])

        return ps

    def improve_hyperpar(self):
        self.result = rinla.inla_hyperpar(self.result)
        return self
