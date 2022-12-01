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
    quantiles: np.ndarray = np.array([0.025, 0.5, 0.975]),
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
    family: str = "gaussian",
    n_trials: Optional[np.ndarray] = None,
    verbose: bool = False,
    only_hyperparam: bool = False,
    inla_call: str = rinla.inla_getOption("inla.call"),
    inla_arg=rinla.inla_getOption("inla.arg"),
    num_threads: int = cpu_count(),
    blas_num_threads: int = 0,
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
    )

    for k in control_params.keys():
        if control_params[k] is None:
            control_params[k] = {}
        control_params[k] = to_list_vector(control_params[k])

    if n_trials is None:
        n_trials = R_NULL

    register_data(data)

    result = rinla.inla(
        formula=ro.r(formula),
        data=convert_py2r(data),
        quantiles=quantiles,
        **control_params,
        family=family,
        Ntrials=n_trials,
        verbose=verbose,
        only_hyperparam=only_hyperparam,
        inla_call=inla_call,
        inla_arg=inla_arg,
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
