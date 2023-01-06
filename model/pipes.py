import math
from os.path import join

import networkx as nx
import numpy as np
import pandas as pd
from IPython import embed
from scipy.spatial import distance

from .constant import External, Junction, Water
from .path import INPUT_DIR
from .utilities import closest_interpolation, shapefile_to_graph, to_dataframe

_water = Water()
_junction = Junction()
_external = External()


class PipesCollection:
    def __init__(
        self,
        shapefile_name: str = "conduite.shp",  # TODO : change default name to pipes.shp
        propreties_filename: str = "data.xlsx",
    ) -> None:
        self.graph = shapefile_to_graph(filename=shapefile_name)
        self.diameters_propreties = to_dataframe(
            filename=propreties_filename,
            directory=INPUT_DIR,
            sheet_name="diameters",
        )
        self.pcs_junctions_propreties = to_dataframe(
            propreties_filename,
            directory=INPUT_DIR,
            sheet_name="pcs_junctions",
        )
        self.pcs_elbows_propreties = to_dataframe(
            propreties_filename,
            directory=INPUT_DIR,
            sheet_name="pcs_elbows",
        )
        self.resume = pd.DataFrame()
        self.junctions_coordinates = set()
        self.junctions = pd.DataFrame()
        self.get_pipe_data()
        self.get_junction()
        self.assign_singular_load_losses()
        self.assing_init_thermal_loss_coefficient()
        self.assing_init_external_temperature()

    def get_pipe_data(self) -> None:
        """
        Retrieve data needed from the shape files.
        """
        # TODO : check if the id is unique
        def get_distance(a_point: tuple, b_point: tuple) -> float:
            """
            Compute de euclidean (cartesian) length between to points in km.
            The point coordinates are given in m (conversion factor is needed).
            """
            return distance.euclidean(a_point, b_point) / 1000

        def get_real_diameter(dn: int) -> float:
            """
            Get real diameter based on the nominal diameter (DN).
            """
            return float(
                self.diameters_propreties[self.diameters_propreties["DN"] == dn][
                    "d int [m]"
                ]
            )

        def get_roughness():
            # TODO : add it in data.xlsx
            return 0.045

        data = {
            "id": list(),
            "DN": list(),
            "from_junction": list(),
            "to_junction": list(),
            "length_km": list(),  # TODO : check units (check CRS)
            # Other parameters to be added :
            "diameter_m": list(),
            "k_mm": list(),
            "loss_coefficient_forward": list(),
            "loss_coefficient_backward": list(),
            "alpha_w_per_m2k": list(),
            "text_k": list(),
            "name": list(),
        }
        pipes_dict = nx.to_dict_of_dicts(self.graph)
        for main_node_coordinates, main_node_data in pipes_dict.items():
            if main_node_data != {}:
                for (
                    neighbourhood_coordinates,
                    neighbourhood_data,
                ) in main_node_data.items():
                    # embed()
                    # exit()
                    data["id"].append(neighbourhood_data.get("id"))
                    dn = neighbourhood_data.get("DN")
                    data["DN"].append(dn)
                    data["from_junction"].append(main_node_coordinates)
                    data["to_junction"].append(neighbourhood_coordinates)
                    length_km = get_distance(
                        a_point=main_node_coordinates,
                        b_point=neighbourhood_coordinates,
                    )
                    data["length_km"].append(length_km)
                    diameter_m = get_real_diameter(dn=dn)
                    data["diameter_m"].append(diameter_m)
                    data["k_mm"].append(get_roughness())
                    data["loss_coefficient_forward"].append(0)
                    data["loss_coefficient_backward"].append(0)
                    data["alpha_w_per_m2k"].append(None)
                    data["text_k"].append(None)
                    name = "pipe_" + str(neighbourhood_data.get("id"))
                    data["name"].append(name)
        self.resume = pd.DataFrame(data=data)
        self.junctions_coordinates = set(pipes_dict.keys())

    def get_junction(self) -> None:
        geodata = list(self.junctions_coordinates)
        dataframe = pd.DataFrame(
            data={
                "geodata": geodata,
                "name": [
                    str(geo_data) for geo_data in geodata
                ],  # TODO : should be : (lg, lat)_junction_{id}
            },
        )
        dataframe["pn_bar"] = _junction.pn_bar
        dataframe["tfluid_k"] = _junction.tfluid_k
        # TODO : add dataframe["height_m"]
        self.junctions = dataframe

    def assign_singular_load_losses(self) -> None:
        """
        Assign singular load losses to the resume dataframe
        for backward and forward directions.
        """

        def get_trygonometic_angle(reference_point: tuple, point: tuple) -> float:
            """
            Calculate the angle between the vector1 (from reference_point and point)
            and the vector2 (horizontal vector).
            """
            if reference_point == point:
                return 0
            else:
                # create vectors
                a = np.array((point[1], point[0]))
                b = np.array((reference_point[1], reference_point[0]))
                c = np.array((reference_point[1] + 100, reference_point[0]))
                vector1 = a - b
                vector2 = c - b
                cosinus = np.dot(vector1, vector2) / (
                    np.linalg.norm(vector1) * np.linalg.norm(vector2)
                )
                # compute angle and convert in degrees
                sign = 1
                if vector1[1] == abs(vector1[1]):
                    sign = -1
                angle = sign * np.arccos(cosinus)
                angle = np.degrees(angle)
                if angle != abs(angle):
                    angle += 360
                # print(f"point = {point} - ba = {vector1} - bc = {vector2} - angle_degree = {angle}")
                return angle

        def get_elbows_singular_load_losses(angles: list) -> float:
            """
            Get singular load losses of an elbow based on the angles between
            two vectors (pipes) and the horizotal line.
            """

            def take_closest(angle: float, angles: list) -> float:
                """
                Find the closest value in a list.
                """
                return min(angles, key=lambda x: abs(x - angle))

            angle = abs(max(angles) - min(angles))
            reference_angles = self.pcs_elbows_propreties.angle.to_list()
            closest_angle = take_closest(angle=angle, angles=reference_angles)
            xi = self.pcs_elbows_propreties.loc[
                self.pcs_elbows_propreties["angle"] == closest_angle, "xi"
            ]
            # embed(header="get_elbows_singular_load_losses")
            return float(xi)

        def get_junction_singular_load_losses(real_ratio: float, ratio: list) -> float:
            ratio = closest_interpolation(items=ratio, value=real_ratio)
            sample = self.pcs_junctions_propreties.loc[
                self.pcs_junctions_propreties.ratio == ratio
            ]
            return (
                float(sample.xi_forward_1),
                float(sample.xi_backward_1),
                float(sample.xi_forward_2),
                float(sample.xi_backward_2),
            )

        def get_two_closest_elements(numbers: list) -> list:
            """
            Find the two closest numbers in a list of three numbers.
            """
            if len(numbers) != 3:
                # embed(header="stop")
                raise NotImplementedError(f"List with the wrong size : {numbers}")
            sorted_numbers = sorted(set(numbers))
            return min(
                [[a, b] for a, b in zip(sorted_numbers, sorted_numbers[1:])],
                key=lambda x: x[1] - x[0],
            )

        # embed(header="*"*30)
        for nodes_id in self.junctions.geodata:
            # embed(header="nodes_id")
            # exit()
            edges = self.resume[
                (self.resume["from_junction"] == nodes_id)
                | (self.resume["to_junction"] == nodes_id)
            ]
            n_nodes = edges.shape[0]
            if n_nodes == 1:  # node that does not join edges
                continue
            elif n_nodes == 2:  # node that joins two edges
                # calculate the angle
                edges["from_trigonometric_angle"] = [
                    get_trygonometic_angle(reference_point=nodes_id, point=tested_node)
                    for tested_node in edges["from_junction"].to_list()
                ]
                edges["to_trigonometric_angle"] = [
                    get_trygonometic_angle(reference_point=nodes_id, point=tested_node)
                    for tested_node in edges["to_junction"].to_list()
                ]
                edges["trigonometric_angle"] = (
                    edges["from_trigonometric_angle"] + edges["to_trigonometric_angle"]
                )
                # find define & define the load losses
                angles = edges.trigonometric_angle.to_list()
                xi = get_elbows_singular_load_losses(angles=angles)
                # assign load losses to a pipe
                self.resume.loc[edges.DN.index.min(), "loss_coefficient_forward"] += xi
                self.resume.loc[edges.DN.index.min(), "loss_coefficient_backward"] += xi
            elif n_nodes == 3:  # node that joins three edges
                # calculate the angle
                edges["from_trigonometric_angle"] = [
                    get_trygonometic_angle(reference_point=nodes_id, point=tested_node)
                    for tested_node in edges["from_junction"].to_list()
                ]
                edges["to_trigonometric_angle"] = [
                    get_trygonometic_angle(reference_point=nodes_id, point=tested_node)
                    for tested_node in edges["to_junction"].to_list()
                ]
                edges["trigonometric_angle"] = (
                    edges["from_trigonometric_angle"] + edges["to_trigonometric_angle"]
                )
                # find the main pipes based on DN
                edges["is_main_pipe_from_DN"] = np.where(
                    edges["DN"] == edges["DN"].max(),
                    True,
                    False,
                )
                #  find the main pipes based on angle
                numbers = edges["trigonometric_angle"].to_list()
                closest_angles = get_two_closest_elements(numbers=numbers)
                edges["is_main_pipe_from_angle"] = edges["trigonometric_angle"].apply(
                    lambda angle: False if angle in closest_angles else True
                )
                # compare the method two find main pipe
                if (
                    edges.is_main_pipe_from_DN.to_list()
                    != edges.is_main_pipe_from_angle.to_list()
                ):
                    msg = "WARNING : No matching between the method to find the main pipes"
                    print(msg)
                if edges.DN.min() != edges.DN.max():
                    real_ratio = edges.DN.min() / edges.DN.max()
                    (
                        xi_forward_1,
                        xi_backward_1,
                        xi_forward_2,
                        xi_backward_2,
                    ) = get_junction_singular_load_losses(
                        real_ratio=real_ratio,
                        ratio=self.pcs_junctions_propreties.ratio.to_list(),
                    )
                    first_min_pipe = edges.loc[edges.DN == edges.DN.min()].index
                    self.resume.loc[
                        first_min_pipe, "loss_coefficient_forward"
                    ] += xi_forward_1
                    self.resume.loc[
                        first_min_pipe, "loss_coefficient_backward"
                    ] += xi_backward_1
                    second_min_pipe = edges.loc[
                        (edges.DN != edges.DN.max()) & (edges.DN != edges.DN.min())
                    ].index
                    self.resume.loc[
                        second_min_pipe, "loss_coefficient_forward"
                    ] += xi_forward_2
                    self.resume.loc[
                        second_min_pipe, "loss_coefficient_backward"
                    ] += xi_backward_2
            elif n_nodes > 3:
                self.resume.loc[edges.index, "loss_coefficient_forward"] += 1.5
                self.resume.loc[edges.index, "loss_coefficient_backward"] += 1.5
            elif n_nodes == 0:

                raise NotImplementedError
            else:
                raise NotImplementedError

    def assing_init_thermal_loss_coefficient(self, isolation_type: int = 1) -> None:
        def give_alpha(
            internal_diameter: float,  # m
            velocity: float = 5,  # m/s
            lam: float = 0.666,
            cp: float = _water.cp,  # Joule/kg K
            eta: float = _water.eta,  # Pa.s
        ):
            prandtl = (eta * cp) / lam
            reynolds = (velocity * internal_diameter * 1000) / eta
            if reynolds <= 10**4:
                nu = 3.66
            elif reynolds > 10**4 and reynolds <= 3 * 10**4:
                nu = 0.0395 * reynolds**0.75 * prandtl ** (1 / 3)
            elif reynolds > 3 * 10**4 and reynolds <= 5 * 10**6:
                nu = 0.023 * reynolds**0.8 * prandtl ** (1 / 3)
            else:
                raise NotImplementedError
            alpha = (nu * lam) / internal_diameter
            return alpha

        def get_u_value(
            nominal_diameter: float,
            isolation_type: int = 1,
            lambd=0.026,
        ):
            De = float(
                self.diameters_propreties.loc[
                    self.diameters_propreties.DN == nominal_diameter,
                    "d ext " + str(isolation_type) + " [m]",
                ]
            )
            Di = float(
                self.diameters_propreties.loc[
                    self.diameters_propreties.DN == nominal_diameter, "d [m]"
                ]
            )
            Dint = float(
                self.diameters_propreties.loc[
                    self.diameters_propreties.DN == nominal_diameter, "d int [m]"
                ]
            )
            alpha = give_alpha(internal_diameter=Dint)
            U = (De / (Di * alpha) + (De * math.log(De / Di)) / (2 * lambd)) ** (-1)
            Ua = U * Di * math.pi
            return round(Ua, 3)

        self.resume["alpha_w_per_m2k"] = self.diameters_propreties["DN"].apply(
            lambda DN: get_u_value(
                nominal_diameter=DN,
                isolation_type=isolation_type,
            )
        )

    def assing_init_external_temperature(self) -> None:
        if "text_k" in self.resume.columns:
            self.resume.text_k = _external.text_k
        else:
            msg = f"This keys does not exist"
            raise KeyError(msg)
        # print("done \n\n")
        pass
