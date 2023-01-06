import pandas as pd
from IPython import embed

from .path import INPUT_DIR
from .utilities import shapefile_to_graph


class SubstationsCollection:
    def __init__(
        self,
        shapefile_name: str = "substations.shp",
        source_filename: str = "substations.xlsx",
    ) -> None:
        self.graph = shapefile_to_graph(filename=shapefile_name)
        self.substation = self.get_substation_data()
        self.resume = pd.DataFrame()
        self.get_resume()

    def get_substation_data(self) -> pd.DataFrame:
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
        # TODO: add pn_bar and tfluid_k in the shapefile attributs table
        self.resume = self.substation.copy()
        self.resume.drop(["shapefile_name"], axis=1, inplace=True)
        self.resume["pn_bar"] = 1.0
        self.resume["tfluid_k"] = 273.15 + 10  # TODO : shape
        self.resume["name"] = [
            f"{geo_data}_valve_{id}"
            for id, geo_data in enumerate(list(self.substation.geodata))
        ]
