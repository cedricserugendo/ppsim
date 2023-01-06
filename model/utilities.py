from datetime import date
from os.path import isfile, join

import fiona
import geopandas as gpd
import networkx as nx
import pandapipes as pp
import pandas as pd
from IPython import embed
from pandapipes.pandapipes_net import pandapipesNet

from .constant import DEFAULT_PROJECTION_EPSG
from .path import INPUT_DIR, NETWORK_BACKUP_DIR

# FILE READING


def shapefile_to_graph(filename: str, **kwargs) -> nx.classes.digraph.DiGraph:
    # TODO : change the projection if needed
    # https://automating-gis-processes.github.io/CSC18/lessons/L2/projections.html
    path = join(INPUT_DIR, filename)
    data = gpd.read_file(path)
    if data.crs.to_epsg() == DEFAULT_PROJECTION_EPSG:
        pass
    elif data.crs.to_dict().get("units") != "m":
        pass
    else:
        raise NotImplementedError
    return nx.read_shp(path, **kwargs)


# INTERPOLATION


def closest_interpolation(items: list, value: float):
    """
    Returns the value from a list closest to a given value.
    """
    closest_item_index = min(
        range(len(items)),
        key=lambda i: abs(items[i] - value),
    )
    return items[closest_item_index]


# DATAFRAME


def to_dataframe(filename: str, directory: str, **kwargs) -> pd.DataFrame:
    """
    Read file (XLSX, CSV) from directory
    and serve it as dataframe.
    """
    path = join(directory, filename)
    if not isfile(path):
        raise FileNotFoundError(f"{path}")
    if filename.endswith(".xlsx"):
        if kwargs.get("sheet_name") is None:
            raise ValueError("Not 'sheet_name' define.")
        dataframe = pd.read_excel(
            path,
            sheet_name=kwargs["sheet_name"],
            engine="openpyxl",
        )
    elif filename.endswith(".csv"):
        dataframe = pd.read_csv(path, **kwargs)
    else:
        raise ValueError("File extension not supported")
    return dataframe


# TIME


def current_datetime() -> str:
    """
    Return the string with current date time.
    Example : December 27, 2022
    """
    today = date.today()
    return today.strftime(r"%Y_%m_%d")


def load_saved_network(filename: str) -> pandapipesNet:
    file = join(NETWORK_BACKUP_DIR, filename)
    if not isfile(file):
        raise FileNotFoundError
    elif file.endswith(".pickle"):
        return pp.from_pickle(file)
    elif file.endswith(".json"):
        return pp.from_json(file)
    else:
        raise ValueError(f"Extension file is not supported")
