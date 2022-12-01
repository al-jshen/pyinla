from typing import Optional
from pyinla.convert import *
from pyinla.utils import base, load_inla
import pandas as pd

rinla = load_inla()


def summary(result):
    """Print a summary of the results."""
    print(base.summary(result))


def inla(
    formula: str,
    data: pd.DataFrame | dict,
    control_compute: Optional[dict] = None,
    control_predictor: Optional[dict] = None,
    family: str = "gaussian",
    Ntrials: Optional[np.ndarray] = None,
    verbose: bool = False,
):
    if isinstance(data, dict):
        data = to_list_vector(data)
    elif isinstance(data, pd.DataFrame):
        data = to_dataframe(data)

    if control_compute is None:
        control_compute = {}
    if control_predictor is None:
        control_predictor = {}

    control_compute = to_list_vector(control_compute)
    control_predictor = to_list_vector(control_predictor)

    return rinla.inla(
        formula=ro.r(formula),
        data=data,
        control_compute=control_compute,
        control_predictor=control_predictor,
        family=family,
        Ntrial=Ntrials,
        verbose=verbose,
    )
