import geopandas as gpd
import matplotlib.pyplot as plt

def plot_map(geo_path):
    gdf = gpd.read_file(geo_path)
    gdf.plot()
    plt.savefig("map.png")
