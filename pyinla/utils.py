import multiprocessing
from typing import Any, Optional

from rpy2 import robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import ListVector

rinla = importr("INLA")
utils = importr("utils")
stats = importr("stats")
base = importr("base")
base.options(Ncpus=multiprocessing.cpu_count())
Package = Any


def inla_summary(result: ListVector) -> str:
    """Return a printable summary of the INLA results."""
    rr = ListVector({k: v for k, v in result.items() if k != "call"})
    return rinla.summary_inla(rr)


def install_inla(testing=True, binary=False) -> None:
    """
    Installs INLA package and dependencies in R.
    """
    print("Make sure to install libgit2, gsl, gdal, udunits2, geos, and proj.")
    _ = input("Press enter to continue.")
    try:
        _ = ro.r(
            f"""
            chooseCRANmirror(graphics=FALSE, ind=1)
            install.packages(c("Matrix", "foreach", "sp", "rgdal", "geoR", "raster"))
            install.packages("BiocManager")
            BiocManager::install(c("graph", "Rgraphviz"), dep=TRUE)
            install.packages("INLA", repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/{'testing' if testing else 'stable'}"), dep=TRUE)
            """
        )
        if binary:
            _ = ro.r(
                """
                library(INLA)
                inla.binary.install()
                """
            )
        print("INLA installed")
    except Exception as e:
        print(e)
        print("INLA is not installed. Please install it manually.")


def set_pardiso_license(
    inla: Package, path: Optional[str] = None, key: Optional[str] = None
) -> None:
    """
    Set PARDISO license file path.
    """
    assert path is not None or key is not None, "Either path or key must be provided."
    if path is not None:
        inla.inla_setOption("pardiso.license", path)
    if key is not None:
        inla.inla_setOption("pardiso.license", key)
    inla.inla_pardiso_check()
