from typing import Optional

from rpy2.robjects.packages import importr

from pyinla.convert import R_NULL
from pyinla.utils import ro

ro.r("library(raster)")
libraster = importr("raster")
raster_extract = ro.r("raster::extract")


def raster_to_points(raster):
    return libraster.rasterToPoints(raster)


def raster_aggregate(
    raster,
    fact: int,
    fun: str = "mean",
    expand: bool = False,
    na_rm: bool = True,
    by: Optional[str | int] = None,
):
    if by is None:
        by = R_NULL

    return ro.r("raster::aggregate")(
        raster, fact, ro.r(fun), expand=expand, na_rm=na_rm, by=by
    )
