from rpy2 import robjects as ro
from rpy2.robjects.vectors import ListVector, DataFrame, StrVector, BoolVector
from rpy2.robjects import pandas2ri
import pandas as pd
import numpy as np

pandas2ri.activate()

R_NULL = ro.rinterface.NULL


def is_null_r(value) -> bool:
    """Checks if an R(py2) value is null."""
    return value == R_NULL


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


def to_dataframe(value: dict) -> DataFrame:
    """Converts a Python dictionary to a DataFrame."""
    return DataFrame(value)


def from_str_vector(str_vector: StrVector) -> str:
    """Converts a StrVector to str."""
    return str(str_vector)


def from_bool_vector(bool_vector: BoolVector) -> bool:
    """Converts a BoolVector to bool."""
    return bool(bool_vector)


def autoconvert(value):
    """Attempt to convert an R object to a Python object."""
    if isinstance(value, ListVector):
        names = value.names
        if is_null_r(names):
            return {}
        return {key: autoconvert(value.rx2(key)) for key in value.names}
    if isinstance(value, dict):
        return {key: autoconvert(value[key]) for key in value}
    if isinstance(value, list):
        return [autoconvert(i) for i in value]
    elif isinstance(value, DataFrame):
        return from_dataframe(value)
    elif isinstance(value, StrVector):
        return from_str_vector(value)
    elif isinstance(value, BoolVector):
        return from_bool_vector(value)
    elif isinstance(value, np.ndarray):
        return value
    elif is_null_r(value):
        return None
    else:
        return value


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
