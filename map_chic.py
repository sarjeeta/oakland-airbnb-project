# -*- coding: utf-8 -*-
"""
Created on Wed May 12 01:51:49 2021

@author: ADI
"""
#importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.express as px
import plotly.io as pio
pio.renderers.default = 'browser'
import urllib.request
from urllib.request import urlopen
import json
import geopandas as gpd

url = 'https://raw.githubusercontent.com/Roopam-kapoor/test/main/Oakland_final.csv'
listing = pd.read_csv(url)
#print(listing.head(2))

map_chic = json.load(open(r'"C:\Users\Roopam Kapoor\OneDrive\Desktop\neighbourhoods_oakland.geojson"', 'r'))
print(listing["neighbourhood"][0])
print(map_chic['features'][0]["properties"])

fig = px.choropleth_mapbox(listing, geojson=map_chic, color="room_type",
                           locations="neighbourhood", featureidkey="properties.neighbourhood",
                           center={"lat": 45.5517, "lon": -73.7073},
                           mapbox_style="carto-positron", zoom=9)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()



