from rpy2 import robjects
from rpy2.robjects.packages import importr
from typing import Any, Optional
import multiprocessing

utils = importr("utils")
base = importr("base")
base.options(Ncpus=multiprocessing.cpu_count())
Package = Any


def install_inla(testing=True) -> None:
    """
    Installs INLA package and dependencies in R.
    """
    print("Make sure to install libgit2, gsl, gdal, udunits2, geos, and proj.")
    _ = input("Press enter to continue.")
    try:
        _ = robjects.r(
            f"""
            chooseCRANmirror(graphics=FALSE, ind=1)
            install.packages(c("Matrix", "foreach", "sp"))
            install.packages("BiocManager")
            BiocManager::install(c("graph", "Rgraphviz"), dep=TRUE)
            install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/{'testing' if testing else 'stable'}"), dep=TRUE)
            """
        )
        print("INLA installed")
    except Exception as e:
        print(e)
        print("INLA is not installed. Please install it manually.")


def load_inla() -> Package:
    inla = importr("INLA")
    return inla


def set_pardiso_license(
    inla: Package, path: Optional[str] = None, key: Optional[str] = None
) -> None:
    """
    Set PARDISO license file path.
    """
    assert path is not None or key is not None, "Either path or key must be provided."
    if path is not None:
        inla.setOption("pardiso.license", path)
    if key is not None:
        inla.setOption("pardiso.license", key)
    inla.pardiso.check()
