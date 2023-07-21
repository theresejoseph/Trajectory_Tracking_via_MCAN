import pandas as pd
from plotnine import *
from geopandas import GeoDataFrame
import folium
import shapely.affinity
from shapely.geometry import Point, LineString
import geopy
import geopy.distance
import random
import geopandas as gpd

import math
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np


def saveRoadMap(shp_path, savePath, center,n):
    roads_berlin = GeoDataFrame.from_file(shp_path, encoding='utf-8')
    roads_berlin.head()

    # create an ellipse
    radius_lat = geopy.distance.distance(kilometers = 5).destination(point=center, bearing=0)[0] - center[0]
    radius_lon = geopy.distance.distance(kilometers = 5).destination(point=center, bearing=90)[1] - center[1]

    circle = Point(center[::-1]).buffer(1, cap_style=3)
    ellipse = shapely.affinity.scale(circle, radius_lon, radius_lat)

    ellipse_map = folium.Map(location=center, zoom_start=10, tiles='CartoDB positron')

    folium.PolyLine(list([c[::-1] for c in ellipse.exterior.coords])).add_to(ellipse_map)

    # roads data
    berlin = roads_berlin.loc[roads_berlin['geometry'].apply(lambda g: ellipse.contains(g))].copy()
    berlin=berlin[berlin['maxspeed']>=n]

    unqiue_speeds=np.sort(berlin.maxspeed.unique())
    print(unqiue_speeds)
    # cmap = plt.get_cmap('tab20c', len(unqiue_speeds) ) # brisbane 
    # cmap = plt.get_cmap('Set2_r', len(unqiue_speeds) ) # berlin
    # cmap = plt.get_cmap('Set1_r', len(unqiue_speeds) ) #japan
    cmap = plt.get_cmap('Paired', len(unqiue_speeds) ) #newyork
    colorMap=[cmap(i) for i in range(len(unqiue_speeds))]

    colors = dict(zip(unqiue_speeds, colorMap))

    top = center[0]+radius_lat
    bottom = center[0]-radius_lat
    left = center[1]-radius_lon
    right = center[1]+radius_lon

    linewidth_lane=0.05
    legend_title= 'Berlin Road Speeds'
    legend_bbox=(1.25,0.5)

    f, ax = plt.subplots(1, figsize=(8, 8))

    a=berlin.plot(ax=ax, linewidth=1,
                color=[colors[d] for d in berlin.maxspeed])
    # a=berlin.plot(ax=ax, linewidth=1, color='k')
    ax.set_aspect(1/math.cos(math.pi/180*center[0]))
    ax.set_ylim((bottom, top))
    ax.set_xlim((left, right))
    # a.set_linewidth(1)
    # a.add_artist(ScaleBar(distance_meters))

    plt.axis('off')
    dpi=300
    
    plt.savefig(savePath, bbox_inches='tight', dpi=dpi)
    # plt.show()

'''berlin'''
# center = [52.520008,13.404954]
# shp_path ='./data/Geospatial/Berlin/gis_osm_roads_free_1.shp' 
# savePath='./results/TestEnvironmentFiles/TestingMaps/berlin_5kmrad_1Line_300pdi.png'
# saveRoadMap(shp_path, savePath, center,5)

'''brisbane'''
# center = [-27.470125, 153.021072]
# shp_path ='./data/Geospatial/Brisbane/gis_osm_roads_free_1.shp' 
# savePath='./results/TestEnvironmentFiles/TestingMaps/brisbane_5kmrad_1Line_300pdi.png'
# saveRoadMap(shp_path, savePath, center,5)

'''japan'''
# center = [35.652832,139.839478]
# shp_path ='./data/Geospatial/Kanto/gis_osm_roads_free_1.shp' 
# savePath='./results/TestEnvironmentFiles/TestingMaps/japan_5kmrad_1Line_300pdi.png'
# saveRoadMap(shp_path, savePath, center,5)

'''newyork'''
# center = [40.730610, -73.935242]
# shp_path ='./data/Geospatial/NewYork/gis_osm_roads_free_1.shp' 
# savePath='./results/TestEnvironmentFiles/TestingMaps/newyork_5kmrad_1Line_300dpi.png'
# saveRoadMap(shp_path, savePath, center,10)