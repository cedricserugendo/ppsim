from IPython import embed
from model.network import Network
from model.pipes import PipesCollection
from model.stations import StationsCollection
from model.substations import SubstationsCollection


def ppsim():
    
    pipes_collection = PipesCollection(
        shapefile_name="pipes_test.shp",
        propreties_filename="pipes.xlsx",
    )

    stations_collection = StationsCollection(
        shapefile_name="sources_test.shp",
        source_filename="sources.xlsx",
    )

    substations_collection = SubstationsCollection(
        shapefile_name="substations_test.shp",
        source_filename="substations.xlsx",
    )
    
    network = Network(
        name="Matran",
        fluid="water",
        pipes_collection=pipes_collection,
        substations_collection=substations_collection,
        stations_collection=stations_collection,
    )
    network.plot_network()
    network.simulate()
    network.plot_pressure_graph()
    network.plot_temperature_graph()
    embed(header="END")


if __name__ == "__main__":
    ppsim()
