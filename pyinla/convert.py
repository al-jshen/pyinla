from copy import deepcopy

import numpy as np
import pandas as pd
from rpy2 import robjects as ro
from rpy2 import rinterface as ri
from rpy2.robjects import default_converter, globalenv, numpy2ri, pandas2ri
from rpy2.robjects.vectors import BoolVector, DataFrame, ListVector, StrVector, Vector
from rpy2.rinterface_lib import na_values
from rpy2.robjects.numpy2ri import converter as numpy2ri_converter

rpy2py = numpy2ri_converter.rpy2py

R_NULL = ro.rinterface.NULL
Converter = default_converter + numpy2ri.converter + pandas2ri.converter

# hacky workaround to deal with NA from R
@rpy2py.register(ri.IntSexpVector)
def rpy2py_intvector(obj):
    x = np.array(obj, dtype=int)
    if np.any(x == -2147483648):
        x = np.array(obj, dtype=float)
        x[np.isclose(x, -2147483648)] = np.nan
    return x


@rpy2py.register(ri.FloatSexpVector)
def rpy2py_floatvector(obj):
    x = np.array(obj)
    x[np.isclose(x, -2147483648)] = np.nan
    return x


numpy2ri.activate()
pandas2ri.activate()


def is_null_r(value) -> bool:
    """Checks if an R(py2) value is null."""
    return value == R_NULL


df_rules = ro.default_converter


@df_rules.rpy2py.register(ri.IntSexpVector)
def to_int(obj):
    return np.array([int(v) if v != na_values.NA_Integer else np.nan for v in obj])


@df_rules.rpy2py.register(ri.FloatSexpVector)
def to_float(obj):
    return np.array([float(v) if v != na_values.NA_Real else np.nan for v in obj])


def pd_to_dict(df):
    """
    Convert a Pandas DataFrame to a dictionary
    """
    return {k: df[k].values for k in df.columns}


def from_list_vector(list_vector: ListVector) -> dict:
    """Converts a ListVector to a Python dictionary."""
    names = list_vector.names
    if is_null_r(names):
        return {}
    return {key: list_vector.rx2(key) for key in list_vector.names}


def from_dataframe(dataframe: DataFrame) -> pd.DataFrame:
    """Converts a DataFrame to a Python dictionary."""
    return pd.DataFrame({key: dataframe.rx2(key) for key in dataframe.names})


def to_list_vector(value: dict) -> ListVector:
    """Converts a Python dictionary to a ListVector."""
    return ListVector(value)


def to_dataframe(value: dict | pd.DataFrame) -> DataFrame:
    """Converts a Python dictionary or Pandas DataFrame to a DataFrame."""
    return DataFrame(value)


def from_str_vector(str_vector: StrVector) -> str:
    """Converts a StrVector to str."""
    return str(str_vector)


def to_str_vector(str: str) -> StrVector:
    """Converts a str to a StrVector."""
    return StrVector(str)


def from_bool_vector(bool_vector: BoolVector) -> bool:
    """Converts a BoolVector to bool."""
    return bool(bool_vector)


def to_bool_vector(value: bool) -> BoolVector:
    """Converts a bool to a BoolVector."""
    return BoolVector(value)


def scalarize(x):
    if not np.isscalar(x) and len(x) == 1:
        return np.asarray(x).item()
    else:
        return x


def convert_r2py(ri):
    """
    Recursively convert rpy2 object to nested Python object.
    Objects containing R code (typeof(x) == 'language') are filtered out.
    Objects containing R lists with no string tags are converted to Python
    lists.
    """
    if isinstance(ri, np.ndarray):
        return ri
    if isinstance(ri, ListVector):
        if ri.names == R_NULL:
            result = [convert_r2py(tmp[1]) for tmp in ri.items()]
        else:
            result = {}
            for name in ri.names:
                globalenv["tmp"] = ri.rx2(name)
                if ro.r("typeof(tmp)")[0] != "language":
                    result[name] = convert_r2py(ri.rx2(name))
        return result
    else:
        if isinstance(ri, Vector) and len(ri) == 0:
            return None
        if ri == R_NULL:
            return None
        res = Converter.rpy2py(ri)
        try:
            return scalarize(res)
        except TypeError:
            return res


def convert_py2r(obj):
    """
    Recursively convert rpy2 object to nested Python object.
    Objects containing R code (typeof(x) == 'language') are filtered out.
    Objects containing R lists with no string tags are converted to Python
    lists.
    """
    try:
        o = deepcopy(obj)
        if isinstance(o, pd.DataFrame):
            o = pd_to_dict(o)
        if isinstance(o, np.ndarray) and np.size(o) == 1 and o == np.nan:
            return na_values.NA_Real
        if isinstance(o, dict):
            if o != {}:
                for k, v in o.items():
                    o[k] = convert_py2r(v)
            return ListVector(o)
        else:
            return Converter.py2rpy(o)
    except:
        print(o)
        raise


def ravel_types(v):
    """
    Find all types in a nested R object
    """
    all_types = set()
    if isinstance(v, ListVector):
        for i in v:
            all_types.update(ravel_types(i))
    else:
        all_types.add(type(v))
    return all_types
