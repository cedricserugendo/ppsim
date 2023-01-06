from os.path import join

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from IPython import embed
from scipy.spatial import distance

from .path import INPUT_DIR
from .utilities import shapefile_to_graph


class StationsCollection:
    def __init__(
        self,
        shapefile_name: str = "sources.shp",
        source_filename: str = "sources.xlsx",
    ) -> None:
        self.graph = shapefile_to_graph(filename=shapefile_name)
        self.production = self.get_production_data(source_filename=source_filename)
        self.source = self.get_source_specifications()
        self.resume = pd.DataFrame()
        self.get_resume()

    def get_production_data(self, source_filename: str) -> pd.DataFrame:
        """
        Create production dataframe.
        """

        def check_revelancy(dataframe: pd.DataFrame, source_ids: list):
            """
            Check revelancy of the production dataframe.
            """
            n_source = len(self.production_source_ids)
            for time_step in set(dataframe["time step"]):
                sample = dataframe.loc[dataframe["time step"] == time_step]
                if not sample.loc[sample["source id"] != source_ids].empty:
                    raise NotImplementedError(
                        f"Source ID is not matching at '{time_step}' time step."
                    )
                if sample["load distribution"].sum() != 1:  # check load distribution
                    raise NotImplementedError
                if n_source != len(
                    sample["source id"].unique()
                ):  # check the number of source
                    raise NotImplementedError

        dataframe = pd.read_excel(
            join(INPUT_DIR, source_filename),
            sheet_name="production",
            engine="openpyxl",
        )
        self.production_source_ids = list(dataframe["source id"].unique())
        check_revelancy(dataframe=dataframe, source_ids=self.production_source_ids)
        return dataframe

    def get_source_specifications(self) -> pd.DataFrame:
        # TODO : change id into source id in the shapefile
        data = dict()
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            node_data["shapefile_name"] = node_data.pop("ShpName")
            node_data["geodata"] = node
            for key, value in node_data.items():
                if data.get(key, None) is None:
                    data[key] = [value]
                else:
                    data[key].append(value)
        return pd.DataFrame(data=data)

    def get_resume(self):
        # TODO : change variable name to match
        self.resume = self.source.copy()
        self.resume["name"] = [f"source_{id}" for id in list(self.resume.id)]
        self.resume.drop(["shapefile_name"], axis=1, inplace=True)
        self.resume["in_service"] = True  # TODO : add value directly in the shapefile
