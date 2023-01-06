import warnings
from os import makedirs
from os.path import isdir, join

import matplotlib.pyplot as plt

import pandapipes as pp
import pandas as pd
from IPython import embed
from pandapower.plotting import cmap_continuous

from .constant import Substation, Water
from .path import NETWORK_BACKUP_DIR
from .pipes import PipesCollection
from .stations import StationsCollection
from .substations import SubstationsCollection
from .utilities import current_datetime
import geopandas as gpd
import tilemapbase
from .constant import DEFAULT_PROJECTION_EPSG
from pandapower.plotting import cmap_discrete, cmap_continuous, create_line_trace, draw_traces

# Ignore pandas warning messages --> issue from the way iterations are made
warnings.simplefilter(action="ignore", category=FutureWarning)

_basic_substation = Substation()


class Network:
    def __init__(
        self,
        name: str,
        pipes_collection: PipesCollection,
        substations_collection: SubstationsCollection,
        stations_collection: StationsCollection,
        fluid: str,
    ) -> None:
        self.name = name
        self.neth = pp.create_empty_network(name=name, fluid=fluid)
        self.pipes = pipes_collection
        self.substations = substations_collection
        self.stations = stations_collection
        self.create_pipes()
        self.create_substations()
        self.create_sources()

    def plot_network(self):
        def create_map():
            ConsPoints = gpd.GeoDataFrame(
                self.neth.junction_geodata,
                geometry=gpd.points_from_xy(
                    self.neth.junction_geodata.x, 
                    self.neth.junction_geodata.y,
                ),
                crs=DEFAULT_PROJECTION_EPSG,
            )
            ConsPoints.to_crs(3857, inplace=True)
            tilemapbase.start_logging()
            tilemapbase.init(create=True)
            extent = tilemapbase.extent_from_frame(ConsPoints, buffer=25)
            bounding_box = [
                self.neth.junction_geodata.x.min(),
                self.neth.junction_geodata.x.max(),
                self.neth.junction_geodata.y.min(),
                self.neth.junction_geodata.y.max(),
            ]
            # WARNING : width changes the quality
            plotter = tilemapbase.Plotter(
                extent, tilemapbase.tiles.build_OSM(), width=1000
            )
            return plotter, bounding_box

        cmap_list_grey = [((0.0, 2), "gray"), ((2, 100), "blue")]
        cmap_grey, norm_grey = cmap_discrete(cmap_list_grey)
        fig_map, axes_map = plt.subplots(nrows=1, ncols=1, figsize=(13, 5))
        # if check_coordinate_types(path):
        plotter, bounding_box = create_map()
        plotter.plot(axes_map, alpha=0.4)
        shift = 25
        axes_map.set_xlim(bounding_box[0] - shift, bounding_box[1] + shift)
        axes_map.set_ylim(bounding_box[2] - shift, bounding_box[3] + shift)
        axes_map.scatter(
            self.neth.junction_geodata.loc[self.neth.heat_exchanger.to_junction, "x"],
            self.neth.junction_geodata.loc[self.neth.heat_exchanger.to_junction, "y"],
            cmap=cmap_grey,
            c=self.neth.heat_exchanger.in_service,
            s=10,
        )
        pip_coll_base = pp.plotting.create_pipe_collection(
            self.neth, use_junction_geodata=True, color="gray",
        )

        pp.plotting.draw_collections(
            (pip_coll_base, pip_coll_base), ax=axes_map, plot_colorbars=True
        )
        axes_map.title.set_text(self.name)
        fig_map.savefig(f"{self.name}_basic_network.png")
        plt.close(fig_map)

    def create_pipes(self):
        """
        Create Pandapipes' pipes.
        First, create the mandatory junction and then
        create the pipes (based on previously created junction).
        """

        def create_pipe_jucntions() -> None:
            def set_pipe_junctions(junction: pd.Series) -> None:
                """
                Set the junctions in the Pandapipes network model.
                Pipe junction :
                    * pn_bar [float]
                    * tfluid_k [float]
                    * name [str] : (long, lat)_pipe_forward/(long, lat)_pipe_backward
                """
                pp.create_junction(
                    self.neth,
                    pn_bar=junction["pn_bar"],
                    tfluid_k=junction["tfluid_k"],
                    name=junction["name"] + "_pipe_forward",
                    geodata=junction["geodata"],
                )
                pp.create_junction(
                    self.neth,
                    pn_bar=junction["pn_bar"],
                    tfluid_k=junction["tfluid_k"],
                    name=junction["name"] + "_pipe_backward",
                    geodata=junction["geodata"],
                )

            self.pipes.junctions.apply(
                lambda junction: set_pipe_junctions(junction=junction),
                axis="columns",
            )

        def set_pipe(pipe: pd.Series) -> None:
            def get_junction_index(direction: str) -> tuple:
                """
                Get the junction index based on fluid flow.
                PIPE JUNCTION NAME : (long, lat)_pipe_{direction} # TODO (general) : specify the name everywhere
                """

                def name_to_index(junction_name: str) -> int:
                    sub_dataframe = self.neth.junction.loc[
                        self.neth.junction.name.str.contains(junction_name, regex=False)
                    ]
                    if sub_dataframe.shape[0] != 1:
                        raise NotImplementedError(
                            "Ambigus name lead to more than one solution"
                        )
                    return sub_dataframe.index[0]

                if direction == "_forward":  # supply
                    from_junction_name = (
                        str(pipe["from_junction"]) + "_pipe" + direction
                    )
                    to_junction_name = str(pipe["to_junction"]) + "_pipe" + direction
                elif direction == "_backward":  # return
                    from_junction_name = str(pipe["to_junction"]) + "_pipe" + direction
                    to_junction_name = str(pipe["from_junction"]) + "_pipe" + direction
                else:
                    raise ValueError("Not supported direction format")
                from_junction_index = name_to_index(junction_name=from_junction_name)
                to_junction_index = name_to_index(junction_name=to_junction_name)
                # print(pipe["id"], from_junction_name, to_junction_name, direction, sep="\t")
                # print(
                #     pipe["id"],
                #     from_junction_index,
                #     to_junction_index,
                #     direction,
                #     sep="\t",
                # )
                return from_junction_index, to_junction_index

            # forward pipe
            from_junction_index, to_junction_index = get_junction_index(
                direction="_forward"
            )
            # print(from_junction_index, to_junction_index)
            pp.create_pipe_from_parameters(
                self.neth,
                from_junction=from_junction_index,
                to_junction=to_junction_index,
                loss_coefficient=pipe["loss_coefficient_forward"],
                length_km=pipe["length_km"],
                diameter_m=pipe["diameter_m"],
                k_mm=pipe["k_mm"],
                alpha_w_per_m2k=pipe["alpha_w_per_m2k"],
                text_k=pipe["text_k"],
                name=pipe["name"] + "_forward",
            )

            # backward pipe
            from_junction_index, to_junction_index = get_junction_index(
                direction="_backward"
            )
            pp.create_pipe_from_parameters(
                self.neth,
                from_junction=from_junction_index,
                to_junction=to_junction_index,
                loss_coefficient=pipe["loss_coefficient_backward"],
                length_km=pipe["length_km"],
                diameter_m=pipe["diameter_m"],
                k_mm=pipe["k_mm"],
                alpha_w_per_m2k=pipe["alpha_w_per_m2k"],
                text_k=pipe["text_k"],
                name=pipe["name"] + "_backward",
            )

        create_pipe_jucntions()
        self.pipes.resume.apply(
            lambda pipe: set_pipe(pipe=pipe),
            axis="columns",
        )

    def create_substations(self) -> None:
        def set_substations_elements(junction: pd.Series) -> None:
            """
            Create junction for the substation.

            Subtatation structure form inpout to the output flow:
                - pip junction
                - valve
                - valve junction
                - heat exchanger
                - pipe junction
            """
            # Create the intermediate junction for the valve

            to_valve = pp.create_junction(
                self.neth,
                pn_bar=junction["pn_bar"],
                tfluid_k=junction["tfluid_k"],
                name=junction["name"],
                geodata=junction["geodata"],
            )

            # create valve
            forward_pipe_name = str(junction["geodata"]) + "_pipe_forward"
            from_junction_exchanger = self.neth.junction.loc[
                forward_pipe_name == self.neth.junction.name
            ].index[0]

            pp.create_valve(
                self.neth,
                from_junction=from_junction_exchanger,
                to_junction=to_valve,
                diameter_m=junction[
                    "diameter_m"
                ],  # TODO : change the name in the shapefile
                name="substation_" + str(junction["id"]),
                index=junction["id"],
                loss_coefficient=_basic_substation.init_loss_coeff,
                opened=_basic_substation.valve_opening,
            )

            # create heat exchanger
            backward_pipe_name = str(junction["geodata"]) + "_pipe_backward"
            to_junction_exchanger = self.neth.junction.loc[
                backward_pipe_name == self.neth.junction.name
            ].index[0]

            pp.create_heat_exchanger(
                self.neth,
                from_junction=to_valve,
                to_junction=to_junction_exchanger,
                diameter_m=junction["diameter_m"],
                qext_w=junction["qext_w"],
                loss_coefficient=1,
                name="substation_" + str(junction["id"]),
                index=junction["id"],
                in_service=True,
                type="heat_exchanger",
                # t_sec=junction["t_sec"] # TODO : add this parameters to the shapefile
            )

        self.substations.resume.apply(
            lambda junction: set_substations_elements(junction=junction),
            axis="columns",
        )

    def create_sources(self) -> None:
        def set_sources_elements(junction: pd.Series) -> None:
            """
            Create junction as well as others elements for sources.

            Sources structure form inpout to the output flow:
                - pipe junction
                - source (input pipe junction)
                - ext grid (on the same junction than source )
                - sink (on the output)
            """
            # input into the grid (w/ specifc mass flow [create_sorurce]
            # and specific pressure and temperature [create_ext_grid])
            forward_pipe_name = str(junction["geodata"]) + "_pipe_forward"
            from_junction_source = self.neth.junction.loc[
                forward_pipe_name == self.neth.junction.name
            ].index[0]
            # print(junction["t_k"])
            # embed()
            source = pp.create_source(
                self.neth,
                junction=from_junction_source,
                mdot_kg_per_s=junction["m_kg_per_s"]
                + 3,  # TODO : change with the volumic density
                t_k=junction["t_k"],  # TODO : change the name in the shapefile
                name="source_" + junction["name"],
                in_service=junction["in_service"],  # TODO : shapefile
            )
            ext_grid = pp.create_ext_grid(
                self.neth,
                junction=from_junction_source,
                p_bar=junction["p_bar"],  # TODO : shapefile
                t_k=junction["t_k"],  # TODO : shapefile
                name="grid_connection_" + junction["name"],
                in_service=junction["in_service"],  # TODO : shapefile
                type="pt",
            )

            # output of the grid (with specific mass flow)
            backward_pipe_name = str(junction["geodata"]) + "_pipe_backward"
            to_junction_source = self.neth.junction.loc[
                backward_pipe_name == self.neth.junction.name
            ].index[0]
            sink = pp.create_sink(
                self.neth,
                junction=to_junction_source,
                mdot_kg_per_s=junction["m_kg_per_s"],  # TODO : volumetric density
                name="sink_" + junction["name"],
                in_service=junction["in_service"],  # TODO : shapefile
            )

        self.stations.resume.apply(
            lambda junction: set_sources_elements(junction=junction),
            axis="columns",
        )

    def simulate(self, iter: int = 500) -> None:
        """
        Step : (for each loop)
        - Lisf the expected mass flows
        - Set mass flow of the sources as the sum of exchanger expected mass flow 
        - Set the mass flow of each exchanger

        - Filter exchanger by minimum power (and set as out of duty)
        - Closed the valve of the exchanger if mandatory

        - Set ext grid and source temperature (equal)
        

        """
        def check_consistency():
            pass

        def calibrate_valve_position():
            pass

        def calibrate_hydraulics_model():
            pp.pipeflow(net=self.neth, mode="hydraulics", iter=iter)
            for time_step in self.stations.production["time step"].unique():
                time_step = str(pd.Timestamp(time_step))
                sbu_dataframe = self.stations.production.loc[
                    self.stations.production["time step"] == time_step
                ]
                # embed(header="CEDRIC SERUGENDO")
                # exit()
            check_consistency()
            calibrate_valve_position()

        calibrate_hydraulics_model()
        pp.pipeflow(net=self.neth, mode="all", iter=iter)

    def plot_pressure_graph(self):
        pipe_results_table = self.neth.res_pipe.copy()
        pipe_results_table["name"] = self.neth.pipe["name"]

        forward_pipe_results_table = pipe_results_table.loc[
            pipe_results_table.name.str.contains("forward")
        ]
        backward_pipe_results_table = pipe_results_table.loc[
            pipe_results_table.name.str.contains("backward")
        ]
        cmap_list = [
            (-0.1, "black"),
            (0, "green"),
            (1.5, "yellow"),
            (3, "red"),
        ]  ## TODO : automatically set the extrema
        cmap_plot, norm = cmap_continuous(cmap_list)
        fig_massflow, (axes_massflow, axes_massflow2) = plt.subplots(
            nrows=2, ncols=2, figsize=(13, 5)
        )
        #####################################################
        base_collection = pp.plotting.create_pipe_collection(
            self.neth,
            use_junction_geodata=True,
            color="gray",
        )

        #####################################################
        forward_from_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=forward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["p_from_bar"],
            cbar_title="p_from_bar",
        )
        pp.plotting.draw_collections(
            (
                base_collection,
                forward_from_collection,
            ),
            ax=axes_massflow[0],
        )
        axes_massflow[0].title.set_text("forward_from_collection")

        forward_to_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=forward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["p_to_bar"],
            cbar_title="p_to_bar",
        )
        pp.plotting.draw_collections(
            (forward_to_collection,),
            ax=axes_massflow[1],
        )
        axes_massflow[1].title.set_text("forward_to_collection")

        ####################################################
        backward_from_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=backward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["p_from_bar"],
            cbar_title="p_from_bar",
        )
        pp.plotting.draw_collections(
            (backward_from_collection,),
            ax=axes_massflow2[0],
        )
        axes_massflow2[0].title.set_text("backward_from_collection")

        backward_to_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=backward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["p_to_bar"],
            cbar_title="p_to_bar",
        )
        pp.plotting.draw_collections(
            (backward_to_collection,),
            ax=axes_massflow2[1],
        )
        axes_massflow2[1].title.set_text("backward_to_collection")
        plt.savefig("pressure.png")
        print("pressure")

    def plot_temperature_graph(self, in_celcius_degree: bool = True):
        pipe_results_table = self.neth.res_pipe.copy()
        pipe_results_table["name"] = self.neth.pipe["name"]

        forward_pipe_results_table = pipe_results_table.loc[
            pipe_results_table.name.str.contains("forward")
        ]
        backward_pipe_results_table = pipe_results_table.loc[
            pipe_results_table.name.str.contains("backward")
        ]
        _OFFSET = 0
        if in_celcius_degree:
            _OFFSET = 273.15

        min_value = self.neth.res_pipe[["t_from_k", "t_to_k"]].min().min() - _OFFSET
        max_value = self.neth.res_pipe[["t_from_k", "t_to_k"]].max().max() - _OFFSET
        mean_value = (min_value + max_value) / 2

        cmap_list = [
            (min_value - 1, "black"),
            (min_value, "green"),
            (mean_value, "yellow"),
            (max_value, "red"),
        ]
        # print(cmap_list)
        # print(cmap_list)
        # print(cmap_list)
        cmap_plot, norm = cmap_continuous(cmap_list)
        fig_massflow, (axes_massflow, axes_massflow2) = plt.subplots(
            nrows=2, ncols=2, figsize=(13, 5)
        )
        #####################################################
        base_collection = pp.plotting.create_pipe_collection(
            self.neth,
            use_junction_geodata=True,
            color="gray",
        )

        #####################################################
        forward_from_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=forward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["t_from_k"] - _OFFSET,
            cbar_title="t_from_k",
        )
        pp.plotting.draw_collections(
            (
                base_collection,
                forward_from_collection,
            ),
            ax=axes_massflow[0],
        )
        axes_massflow[0].title.set_text("forward_from_collection")

        forward_to_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=forward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["t_to_k"] - _OFFSET,
            cbar_title="t_to_k",
        )
        pp.plotting.draw_collections(
            (forward_to_collection,),
            ax=axes_massflow[1],
        )
        axes_massflow[1].title.set_text("forward_to_collection")

        ####################################################
        backward_from_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=backward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["t_from_k"] - _OFFSET,
            cbar_title="t_from_k",
        )
        pp.plotting.draw_collections(
            (backward_from_collection,),
            ax=axes_massflow2[0],
        )
        axes_massflow2[0].title.set_text("backward_from_collection")

        backward_to_collection = pp.plotting.create_pipe_collection(
            self.neth,
            pipes=backward_pipe_results_table.index,
            use_junction_geodata=True,
            cmap=cmap_plot,
            norm=norm,
            z=self.neth.res_pipe["t_to_k"] - _OFFSET,
            cbar_title="t_to_k",
        )
        pp.plotting.draw_collections(
            (backward_to_collection,),
            ax=axes_massflow2[1],
        )
        axes_massflow2[1].title.set_text("backward_to_collection")
        plt.savefig("temperature.png")
        print("temperature")

    def save(self, mode: str = "pickle"):
        saving_file = join(NETWORK_BACKUP_DIR, f"{self.name}_{current_datetime()}")
        if mode == "pickle":
            saving_file += ".pickle"
            pp.to_pickle(self.neth)
        elif mode == "json":
            saving_file += ".json"
            pp.to_json(self.neth)
        else:
            raise ValueError(f"Saving mode not supported : {mode}")
