from rpy2.robjects.packages import InstalledPackage, InstalledSTPackage, importr
from rpy2.robjects.vectors import StrVector


def install_inla() -> None:
    """
    Installs INLA package and dependencies in R.
    """
    try:
        utils = importr("utils")
        dependencies = StrVector(["foreach", "sp"])
        utils.chooseCRANmirror(ind=1)
        utils.install_packages(dependencies)
        utils.install_packages(
            "INLA", repos="https://inla.r-inla-download.org/R/stable", dependencies=True
        )
        print("INLA installed")
    except Exception as e:
        print(e)
        print("INLA is not installed. Please install it manually.")


def load_inla() -> InstalledSTPackage | InstalledPackage:
    inla = importr("INLA")
    return inla
