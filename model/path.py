# PATHS
from os.path import abspath, dirname, isfile, join

CURRENT_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(CURRENT_DIR)
INPUT_DIR = join(ROOT_DIR, "Input")
NETWORK_BACKUP_DIR = join(ROOT_DIR, "network backup")

# TODO : used this path as default value
# SOURCES_SHAPEFILE_PATH = join(INPUT_DIR, "source.shp")
# CONSUMERS_SHAPEFILE_PATH = join(INPUT_DIR, "substations.shp")
# PIPES_SHAPEFILE_PATH = join(INPUT_DIR, "pipes.shp")

# for file in [SOURCES_SHAPEFILE_PATH, CONSUMERS_SHAPEFILE_PATH, PIPES_SHAPEFILE_PATH]:
#     if not isfile(file):
#         raise FileNotFoundError(file)
