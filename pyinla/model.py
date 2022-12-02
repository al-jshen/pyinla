from typing import Optional
from rpy2.robjects import globalenv
from pyinla.convert import *
from pyinla.utils import *
from multiprocessing import cpu_count

rinla = load_inla()


def summary(result: ListVector) -> str:
    """Return a printable summary of the INLA results."""
    rr = ListVector({k: v for k, v in result.items() if k != "call"})
    return rinla.summary_inla(rr)


def register_data(data: dict):
    """Register data in the global environment."""
    with localconverter(Converter):
        for k, v in data.items():
            if isinstance(v, dict):
                register_data(v)
            globalenv[k] = convert_py2r(v)


def inla(
    formula: str,
    data: dict,
    family: str = "gaussian",
    quantiles: np.ndarray = np.array([0.025, 0.5, 0.975]),
    E: Optional[np.ndarray] = None,
    scale: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    n_trials: Optional[np.ndarray] = None,
    control_compute: Optional[dict] = None,
    control_predictor: Optional[dict] = None,
    control_family: Optional[dict] = None,
    control_inla: Optional[dict] = None,
    control_fixed: Optional[dict] = None,
    control_mode: Optional[dict] = None,
    control_expert: Optional[dict] = None,
    control_hazard: Optional[dict] = None,
    control_lincomb: Optional[dict] = None,
    control_update: Optional[dict] = None,
    control_lp_scale: Optional[dict] = None,
    control_pardiso: Optional[dict] = None,
    verbose: bool = False,
    only_hyperparam: bool = False,
    num_threads: int = cpu_count(),
    blas_num_threads: int = cpu_count(),
    keep: bool = False,
    working_directory: str = rinla.inla_getOption("working.directory"),
    silent: bool = True,
    inla_mode: str = rinla.inla_getOption("inla.mode"),
    safe: bool = True,
    debug: bool = False,
):

    control_params = dict(
        control_compute=control_compute,
        control_predictor=control_predictor,
        control_family=control_family,
        control_inla=control_inla,
        control_fixed=control_fixed,
        control_mode=control_mode,
        control_expert=control_expert,
        control_hazard=control_hazard,
        control_lincomb=control_lincomb,
        control_update=control_update,
        control_lp_scale=control_lp_scale,
        control_pardiso=control_pardiso,
    )

    for k in control_params.keys():
        if control_params[k] is None:
            control_params[k] = {}
        control_params[k] = to_list_vector(control_params[k])

    if n_trials is None:
        n_trials = R_NULL

    if E is None:
        E = R_NULL

    if scale is None:
        scale = R_NULL

    if weights is None:
        weights = R_NULL

    register_data(data)

    result = rinla.inla(
        formula=ro.r(formula),
        data=convert_py2r(data),
        family=family,
        quantiles=quantiles,
        E=E,
        scale=scale,
        Ntrials=n_trials,
        weights=weights,
        **control_params,
        verbose=verbose,
        only_hyperparam=only_hyperparam,
        num_threads=num_threads,
        blas_num_threads=blas_num_threads,
        keep=keep,
        working_directory=working_directory,
        silent=silent,
        inla_mode=inla_mode,
        safe=safe,
        debug=debug,
    )

    return result
