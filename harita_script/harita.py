import unidecode
import requests as req
from bs4 import BeautifulSoup
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from sys import is_finalizing
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import requests as req
from bs4 import BeautifulSoup
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import cascaded_union, unary_union, split
from pathlib import Path
import os
import contextily as ctx
from geovoronoi import voronoi_regions_from_coords, points_to_coords
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient





current_working_directory = os.getcwd()
cwd = Path(current_working_directory)
           

def normalize_text(text):
    text = unidecode.unidecode(text)
    text = text.lower()
    text = text.strip()
    text = " ".join(text.split())
    # Gerekiyorsa başka temizlik işlemleri burada yapılabilir
    return text.title()




# Garenta şubeleri sayfasının URL'si
base_url = 'https://www.garenta.com.tr/garenta-subeleri/'
response = req.get(base_url)
soup = BeautifulSoup(response.content, 'html.parser')

# Şube isimleri, adresleri, e-posta adresleri ve telefon numaralarını çekme, html parsing
div_elements = soup.find_all('div', class_='col-lg-9 col-sm-9 col-xs-9 off_detail_box')
office_names=[]
for i, each in enumerate(div_elements, start=1):
    office_name = each.find('h6').text.strip()
    address = each.find('strong', string='Adres:').next_sibling.strip()
    email_tag = each.find('a', href=True)
    email = email_tag['href'] if email_tag else 'Email bilgisi yok'
    phone_number = each.find('strong', string='Telefon:').next_sibling.strip()
    office_names.append(office_name)
    #print(f"Adres: {address}")
    #print(f"Email: {email}")
    #print(f"Telefon: {phone_number}")
    #print("-" * 50)
#print(office_names)
div_elements = soup.find_all('a', class_='clrred off_map mapIco clearfix')
i=0
coordinates=[]
for each in div_elements:
  i=i+1
  latitude = each.get('data-latitude')
  longitude = each.get('data-longitude')
  coordinate = (latitude, longitude)  # Storing as a tuple
  coordinates.append(coordinate)
#print(coordinates)

data = []
for label, (lat, lon) in zip(office_names, coordinates):
    data.append([normalize_text(label), float(lon), float(lat)])

garenta_branches_as_point = pd.DataFrame(data, columns=['Label', 'Longitude', 'Latitude'])

#garenta_branches_as_point.loc[garenta_branches_as_point['Label'] == "ordu giresun havalimanı", 'Label'] = "ordu-giresun havalimanı" #prevent disperancies
garenta_branches_as_point.loc[garenta_branches_as_point['Label'] == "Van Havalimani (Karsilama)", 'Label'] = "Van Sehir" #prevent disprencies
garenta_branches_as_point.loc[garenta_branches_as_point['Label'] == "Istanbul Sabiha Gokcen Hvl. Dis Ht.", 'Label'] = "Istanbul Sabiha Gokcen Hvl." #prevent disperencies
garenta_branches_as_point.loc[garenta_branches_as_point['Label'] == "Istanbul Sabiha Gokcen Hvl. Ic Ht.", 'Label'] = "Istanbul Sabiha Gokcen Hvl."
#garenta_branches_as_point.loc[garenta_branches_as_point['Label'] == "afyonkarahisar park afyon avm	", 'Label'] = "afyonkarahisar - park afyon avm"
garenta_branches_as_point=garenta_branches_as_point.drop_duplicates() # remove one of the sabiha gökçen hvl.

garenta_branches_as_point['City'] = garenta_branches_as_point['Label'].apply(lambda x: x.split("-")[0] if "-" in x else x.split()[0])
garenta_branches_as_point.loc[garenta_branches_as_point['Label'] == "Gazipasa Alanya Havalimani", 'City'] = "Antalya" #ilk isim fonksiyonunda şehir(şehir:"alanya") olarak ihlal eden alanya şubelerin şehirleri antalya olarak düzelti
garenta_branches_as_point.loc[garenta_branches_as_point['Label'] == "Alanya Sehir", 'City'] = "Antalya"


garenta_branches_as_point.to_excel("garenta_branches.xlsx", index=False)

geometry = [Point(xy) for xy in zip(garenta_branches_as_point['Longitude'], garenta_branches_as_point['Latitude'])]
crs = 'epsg:4326' #uygun kordinat sistemi versiyonu
garenta_branchs_as_point_geojson = gpd.GeoDataFrame(garenta_branches_as_point, crs=crs, geometry=geometry)
garenta_branches_point_geojson=garenta_branchs_as_point_geojson[["Label","geometry"]]
garenta_branches_point_geojson.to_file("garenta_city_branches_points.geojson", driver='GeoJSON')



def save_geojson(gdf, output_path, filename):
    full_path = os.path.join(output_path, filename)
    gdf.to_file(full_path, driver='GeoJSON')

def graph_multiple_points(branches_gdf, city_boundaries):
    branches_gdf = gpd.GeoDataFrame(branches_gdf, geometry=gpd.points_from_xy(branches_gdf['Longitude'], branches_gdf['Latitude']), crs='EPSG:4326')
    city_boundaries = city_boundaries.to_crs('EPSG:4326')
    coords = points_to_coords(branches_gdf.geometry)
    city_boundaries = city_boundaries.buffer(0.0001).buffer(-0.0001)
    boundary_shape = unary_union(city_boundaries.geometry)
    poly_shapes, pts, poly_to_pt_assignments = voronoi_regions_from_coords(coords, boundary_shape, return_unassigned_points=True)
    regions_gdf=gpd.GeoDataFrame(geometry=list(poly_shapes.values()),crs='epsg:4326')

    return regions_gdf

def analyze_branch_distribution(branches_gdf):
    std_lat = branches_gdf['Latitude'].std()
    std_lon = branches_gdf['Longitude'].std()
    return 'horizontal' if std_lat > std_lon else 'vertical'

def horizontal_split(city_data, city_boundaries):
    sorted_points = city_data.sort_values(by='Latitude', ascending=False)
    first_point,middle_point,last_point = sorted_points.iloc[0], sorted_points.iloc[1], sorted_points.iloc[-1]
    first_line_lat = first_point['Latitude'] - 1/3*(first_point['Latitude'] - middle_point["Latitude"])
    second_line_lat = last_point['Latitude'] + 1/3 *(middle_point["Latitude"]-last_point['Latitude'])

    regions = define_regions_horizontal(city_boundaries.total_bounds, first_line_lat, second_line_lat)
    return create_regions_gdf(regions, city_boundaries)

def vertical_split(city_data, city_boundaries):
    sorted_points = city_data.sort_values(by='Longitude', ascending=True)
    first_point, last_point = sorted_points.iloc[0], sorted_points.iloc[-1]
    first_line_lon = first_point['Longitude'] + (last_point['Longitude'] - first_point['Longitude']) / 3
    second_line_lon = first_point['Longitude'] + 2 * (last_point['Longitude'] - first_point['Longitude']) / 3

    regions = define_regions_vertical(city_boundaries.total_bounds, first_line_lon, second_line_lon)
    return create_regions_gdf(regions, city_boundaries)

def define_regions_horizontal(bounds, first_line_lat, second_line_lat):
    minx, miny, maxx, maxy = bounds
    region_1 = Polygon([(minx, maxy), (maxx, maxy), (maxx, first_line_lat), (minx, first_line_lat)])
    region_2 = Polygon([(minx, first_line_lat), (maxx, first_line_lat), (maxx, second_line_lat), (minx, second_line_lat)])
    region_3 = Polygon([(minx, second_line_lat), (maxx, second_line_lat), (maxx, miny), (minx, miny)])
    return [region_1, region_2, region_3]

def define_regions_vertical(bounds, first_line_lon, second_line_lon):
    minx, miny, maxx, maxy = bounds
    region_1 = Polygon([(minx, miny), (first_line_lon, miny), (first_line_lon, maxy), (minx, maxy)])
    region_2 = Polygon([(first_line_lon, miny), (second_line_lon, miny), (second_line_lon, maxy), (first_line_lon, maxy)])
    region_3 = Polygon([(second_line_lon, miny), (maxx, miny), (maxx, maxy), (second_line_lon, maxy)])
    return [region_1, region_2, region_3]

def create_regions_gdf(regions, city_boundaries):
    city_union = city_boundaries.unary_union
    clipped_regions = [region.intersection(city_union) for region in regions]
    return gpd.GeoDataFrame({'Label': ['Region 1', 'Region 2', 'Region 3'], 'geometry': clipped_regions}, crs='epsg:4326')

def split_city_into_three_regions(branches_gdf, city_boundaries):
    direction = analyze_branch_distribution(branches_gdf)
    if direction == 'horizontal':
        regions_gdf = horizontal_split(branches_gdf, city_boundaries)
    else:
        regions_gdf = vertical_split(branches_gdf, city_boundaries)
    return regions_gdf
def split_city_into_two(city_data, city_boundary_gdf):

    point1_coords = city_data.iloc[0][['Longitude', 'Latitude']].values
    point2_coords = city_data.iloc[1][['Longitude', 'Latitude']].values
    point1 = Point(point1_coords)
    point2 = Point(point2_coords)

    mid_point = Point((point1.x + point2.x) / 2, (point1.y + point2.y) / 2)
    slope = (point2.y - point1.y) / (point2.x - point1.x)

    if slope != 0:
        perp_slope = -1 / slope
    else:
        perp_slope = np.inf

    if perp_slope != np.inf:
        y_intercept = mid_point.y - (perp_slope * mid_point.x)
        x_values = np.array([point1.x - 2, point2.x + 2])
        y_values = perp_slope * x_values + y_intercept
    else:
        x_values = np.array([mid_point.x, mid_point.x])
        y_values = np.array([point1.y - 0.1, point2.y + 0.1])

    dividing_line = LineString(np.column_stack((x_values, y_values)))

    polygon = city_boundary_gdf.unary_union 

    split_polygons = split(polygon, dividing_line)
    split_gdf = gpd.GeoDataFrame(geometry=list(split_polygons.geoms), crs='epsg:4326')
    #split_gdf.to_file("SON_iki", driver='GeoJSON')

    return split_gdf

def merge_geojsons(output_path):
    files = [os.path.join(output_path, f) for f in os.listdir(output_path) if f.endswith('.geojson')]
    gdfs = [gpd.read_file(f) for f in files]
    merged_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    return merged_gdf


def process_branches(city_name, branches_gdf, city_boundaries_gdf, output_path):
    branch_count = len(branches_gdf)


    if branch_count == 2:
        split_gdf = split_city_into_two(branches_gdf, city_boundaries_gdf)
        save_geojson(split_gdf, output_path, f"{city_name}_split.geojson")
    elif branch_count == 3:
        split_gdf = split_city_into_three_regions(branches_gdf, city_boundaries_gdf)
        save_geojson(split_gdf, output_path, f"{city_name}_triple_split.geojson")
    elif branch_count >= 4:
        voronoi_gdf = graph_multiple_points(branches_gdf, city_boundaries_gdf)
        save_geojson(voronoi_gdf, output_path, f"{city_name}_voronoi.geojson")
    else:
        save_geojson(city_boundaries_gdf, output_path, f"{city_name}_boundary.geojson")

def process_all_cities(branches_gdf, city_boundaries, output_dir):
  output_dir_path = Path(output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)

  # Iterate over each city in Turkey

  for city in city_boundaries['name'].to_list():
      city_data = branches_gdf[branches_gdf['City'].str.lower() == city.lower()]
      city_boundary = city_boundaries[city_boundaries['name'] == normalize_text(city)]

      if not city_boundary.empty:
          process_branches(city, city_data, city_boundary, output_dir_path)
      else:
          print(f"No boundary data found for {city}. Skipping...")

  # Merge all generated GeoJSONs into a single map
  merged_gdf = merge_geojsons(output_dir_path)
  merged_geojson_path = "garenta_haritası.geojson"
  merged_gdf.to_file(merged_geojson_path, driver='GeoJSON')
  #print(f"Merged map saved to {merged_geojson_path}")

branches_dataset = pd.read_excel("garenta_branches.xlsx")
city_boundaries = gpd.read_file("türkiye_haritası.json")
city_boundaries["name"] = city_boundaries["name"].apply(lambda row: normalize_text(row))


output_dir = cwd / 'garenta_görselleştirme'
output_dir.mkdir(exist_ok=True)

process_all_cities(branches_dataset, city_boundaries, output_dir)

merged_map_path = "garenta_haritası.geojson"
city_branches_path = 'garenta_city_branches_points.geojson'

merged_map_gdf = gpd.read_file(merged_map_path)
city_branches_gdf = gpd.read_file(city_branches_path)
count=0
for i, branch in city_branches_gdf.iterrows():
    point = branch.geometry
    flag=True
    for j, polygon in merged_map_gdf.iterrows():
        if point.within(polygon.geometry):# Her bir poligon için noktanın içinde olup olmadığını kontrol et
            merged_map_gdf.at[j, 'Label'] = branch['Label'] # Nokta poligonun içindeyse, poligonun etiketini güncelle
            flag=False
            break

count=0
for i,row in branches_dataset.iterrows():
  if row['Label'] not in merged_map_gdf["Label"].to_list():
      count+=1
      print("LABEL LANAMAYAN YERLER "+ str(count) +" "+ row['Label'])

for i,row in merged_map_gdf.iterrows():
  if row['Label'] == None:
    merged_map_gdf.at[i, "Label"] = row['name']
    

merged_map_gdf = merged_map_gdf[(merged_map_gdf["Label"].notnull()) | (merged_map_gdf["name"].notnull())]

merged_map_gdf.to_file(merged_map_path, driver='GeoJSON')

# Azure Storage hesabınıza ait connection string
conn_string = "DefaultEndpointsProtocol=https;AccountName=sftpdeneme;AccountKey=Ks/pBLXYECIqDTZUa9zATbahogkLAEiGFog2xc41S9YJ4Y6oiOL977t7IqKr+0+UbHfoqdIpI++4+AStX8APEw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(conn_string)
container_name = 'subegorsellestirme'
container_client = blob_service_client.create_container(container_name)

blob_name = 'garenta_haritası.geojson'
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
with open(merged_map_path, "rb") as data:
    blob_client.upload_blob(data, overwrite=True)
