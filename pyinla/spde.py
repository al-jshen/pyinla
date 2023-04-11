from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.methods import RS4
from rpy2.robjects.vectors import FloatVector, ListVector

from pyinla.convert import R_NULL, convert_py2r, convert_r2py, df_rules
from pyinla.utils import base, rinla


class Mesh:
    def __init__(self, inla_mesh):
        self.mesh = inla_mesh
        self.n = self.mesh.rx2("n")
        self.degree = self.mesh.rx2("degree")
        self.interval = self.mesh.rx2("interval")
        self.loc = self.mesh.rx2("loc")
        self.points_idx = self.mesh.rx2("idx").rx2("loc")

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        pass


class Mesh2D(Mesh):
    def __init__(self, inla_mesh):
        super().__init__(inla_mesh)
        self.loc = self.mesh.rx2("loc")[:, :2]

    def plot(self, ax: Optional[plt.Axes] = None, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(**kwargs)
        ax.triplot(self.loc[:, 0], self.loc[:, 1], c="k")
        ax.scatter(
            self.loc[self.points_idx, 0], self.loc[self.points_idx, 1], c="r", s=10
        )
        if ax is None:
            return fig, ax
        return ax


class SPDE2:
    def __init__(self, inla_spde2, mesh_dim: int):
        self.spde = inla_spde2
        self.n_spde = self.spde.rx2("n.spde")
        self.n_theta = self.spde.rx2("n.theta")
        if mesh_dim == 1:
            self.mesh = Mesh(self.spde.rx2("mesh"))
        elif mesh_dim == 2:
            self.mesh = Mesh2D(self.spde.rx2("mesh"))

    def make_index(self, name: str, n_group: int = 1, n_repl: int = 1):
        return rinla.inla_spde_make_index(
            name, self.n_spde, n_group=n_group, n_repl=n_repl
        )


def make_mesh_projector(
    mesh: Mesh,
    loc: Optional[np.ndarray] = None,
    xlims: Optional[List[float]] = None,
    ylims: Optional[List[float]] = None,
    dims: Optional[List[float]] = None,
):
    if loc is None:
        assert (
            xlims is not None and ylims is not None and dims is not None
        ), "if loc is none, must specify xlim, ylim, and dims"
        return rinla.inla_mesh_projector(
            mesh=mesh.mesh,
            xlim=np.array(xlims),
            ylim=np.array(ylims),
            dims=np.array(dims),
        )
    else:
        return rinla.inla_mesh_projector(mesh=mesh.mesh, loc=loc)


def project_mesh(
    projector: ListVector,
    field: np.ndarray,
):
    return rinla.inla_mesh_project(projector, field)


def make_projection_matrix(
    mesh: Mesh,
    loc: np.ndarray | pd.DataFrame,
    group: Optional[np.ndarray] = None,
):
    if isinstance(loc, pd.DataFrame):
        loc = loc.values
    if group is None:
        group = R_NULL
    return rinla.inla_spde_make_A(mesh=mesh.mesh, loc=loc, group=group)


def mesh_2d(
    coords, max_edge: List[float] | float = 1.0, cutoff: Optional[float] = None
):
    """
    x: longitude
    y: latitude
    """
    if cutoff is None:
        cutoff = R_NULL
    return Mesh2D(rinla.inla_mesh_2d(loc=coords, max_edge=max_edge, cutoff=cutoff))


def mesh_1d(
    loc: np.ndarray,
    interval: Optional[List[float]] = None,
    boundary: Optional[str] = None,
    degree: int = 1,
):
    assert degree in [0, 1, 2], "degree must be 0, 1, or 2"
    if interval is None:
        interval = np.array([np.min(loc), np.max(loc)])
    if boundary is None:
        boundary = R_NULL
    else:
        assert boundary in [
            "neumann",
            "dirichlet",
            "free",
            "cyclic",
        ], "boundary must be one of: 'neumann', 'dirichlet', 'free', 'cyclic'"

    return Mesh(
        rinla.inla_mesh_1d(loc=loc, interval=interval, boundary=boundary, degree=degree)
    )


def spde2_pcmatern(
    mesh: Mesh,
    prior_range: List[float] | float | int,
    prior_sigma: List[float] | float | int,
    alpha: float = 2,
    constr: bool = False,
    fractional_method: str = "parsimonious",
    n_iid_group: int = 1,
):
    """
    `prior_range` has values [range0 and Prange], such that the resulting prior has the property P(range < range0) = Prange.
    For example, prior_range = [1, 0.05] means that the prior has 5% probability of being less than 1.
    The user should choose which ranges are too short to be unfeasible for the problem.
    As a heuristic, ranges that are smaller than the resolution of the mesh should be disfavoured.
    Larger values of range0 will result in smoother functions.

    `prior_sigma` has values [sigma0 and Psigma], such that the resulting prior has the property P(sigma > sigma0) = Psigma.
    The user should choose what standard deviations are too large to be feasible for the problem.
    It is probably better to err on the side of a larger sigma0 when unsure.
    """
    assert fractional_method in [
        "parsimonious",
        "null",
    ], "fractional_method must be parsimonious or null"
    assert 0 < alpha <= 2, "alpha must be in (0, 2]"

    if isinstance(prior_range, (int, float)):
        prior_range = [prior_range, np.nan]
    else:
        assert (
            len(prior_range) == 2
        ), "prior_range must be a list of length 2 or a float"

    if isinstance(prior_sigma, (int, float)):
        prior_sigma = [prior_sigma, np.nan]
        assert (
            len(prior_range) == 2
        ), "prior_sigma must be a list of length 2 or a float"

    mesh_dim = 2 if isinstance(mesh, Mesh2D) else 1

    return SPDE2(
        rinla.inla_spde2_pcmatern(
            mesh.mesh,
            alpha=alpha,
            constr=constr,
            fractional_method=fractional_method,
            n_iid_group=n_iid_group,
            prior_range=FloatVector(prior_range),
            prior_sigma=FloatVector(prior_sigma),
        ),
        mesh_dim,
    )


def spde2_matern(
    mesh: Mesh,
    alpha: float = 2,
    constr: bool = False,
    fractional_method: str = "parsimonious",
    B_tau: np.ndarray = np.array([[0, 1, 0]]),
    B_kappa: np.ndarray = np.array([[0, 0, 1]]),
    prior_variance_nominal: float = 1.0,
    prior_range_nominal: Optional[float] = None,
    prior_tau: Optional[float] = None,
    prior_kappa: Optional[float] = None,
    theta_prior_mean: Optional[float] = None,
    theta_prior_prec: Optional[float] = 0.1,
    n_iid_group: int = 1,
):

    assert fractional_method in [
        "parsimonious",
        "null",
    ], "fractional_method must be parsimonious or null"
    assert 0 < alpha <= 2, "alpha must be in (0, 2]"

    if prior_range_nominal is None:
        prior_range_nominal = R_NULL

    if prior_tau is None:
        prior_tau = R_NULL

    if prior_kappa is None:
        prior_kappa = R_NULL

    if theta_prior_mean is None:
        theta_prior_mean = R_NULL

    mesh_dim = 2 if isinstance(mesh, Mesh2D) else 1

    return SPDE2(
        rinla.inla_spde2_matern(
            mesh.mesh,
            alpha=alpha,
            constr=constr,
            fractional_method=fractional_method,
            B_tau=B_tau,
            B_kappa=B_kappa,
            prior_variance_nominal=prior_variance_nominal,
            prior_range_nominal=prior_range_nominal,
            prior_tau=prior_tau,
            prior_kappa=prior_kappa,
            theta_prior_mean=theta_prior_mean,
            theta_prior_prec=theta_prior_prec,
            n_iid_group=n_iid_group,
        ),
        mesh_dim,
    )


def inla_stack(
    tag: str,
    data: dict,
    A: RS4,
    effects: dict,
    index: Optional[ListVector] = None,
):
    if index is None:
        if len(effects.keys()) == 0:
            effects = base.list()
        else:
            effects = base.list(convert_py2r(effects))
    else:
        if len(effects.keys()) == 0:
            effects = base.list(s=index)
        else:
            effects = base.list(convert_py2r(effects), s=index)
    len_effects = base.length(effects)
    with localconverter(df_rules):
        return rinla.inla_stack(
            data=convert_py2r(data),
            A=convert_py2r([1] * int(len_effects - 1) + [A]),
            effects=effects,
            tag=tag,
            compress=True,
        )


def combine_stacks(*stacks):
    with localconverter(df_rules):
        return rinla.inla_stack(*stacks)


def stack_data(stack):
    with localconverter(df_rules):
        stacked = rinla.inla_stack_data(stack)
    return convert_r2py(stacked)


def stack_A(stack):
    return rinla.inla_stack_A(stack)


def stack_index(stack, tag):
    return rinla.inla_stack_index(stack, tag).rx2("data").astype(int)
