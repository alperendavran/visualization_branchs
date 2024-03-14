from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import folium
import json
import os
from tempfile import mkdtemp
import numpy as np
import unidecode
import folium.plugins as plugins
from io import BytesIO
from folium.plugins import MarkerCluster
from azure.storage.blob import BlobServiceClient



app = Flask(__name__)


def normalize_text(text):
    text = unidecode.unidecode(text)
    text = text.lower()
    text = text.strip()
    text = " ".join(text.split())
    # Gerekiyorsa başka temizlik işlemleri burada yapılabilir
    return text.title()


conn_string = "DefaultEndpointsProtocol=https;AccountName=sftpdeneme;AccountKey=Ks/pBLXYECIqDTZUa9zATbahogkLAEiGFog2xc41S9YJ4Y6oiOL977t7IqKr+0+UbHfoqdIpI++4+AStX8APEw==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(conn_string)
container_name = 'subegorsellestirme'
blob_client = blob_service_client.get_blob_client(container = container_name, blob="garenta_haritası.geojson")


with open("garenta_haritası.geojson", "wb") as download_file:
    download_file.write(blob_client.download_blob().readall())

with open("garenta_haritası.geojson", "r") as file:
    garenta_map = json.load(file)

for feature in garenta_map['features']:
    if 'Label' not in feature['properties'] or feature['properties']['Label'] is None:
      feature['properties']['Label'] = 'Placeholder'
    feature['properties']['Label'] = normalize_text(feature['properties']['Label'])

def totaly_sure_percent(df, capacity_column):
    for i, row in df.iterrows():
        if row[capacity_column]!="":
            if row[capacity_column] > 1:
                return False
    return True

    
def adjust_threshold_scale(data, column, manual_thresholds):
    min_value = data[column].min()
    max_value = data[column].max()

    # Kullanıcı tarafından girilen eşik değerler listesinin ilk elemanı veri setinin minimum değerinden büyükse, veri setinin minimum değerini listenin başına ekle.
    if manual_thresholds and manual_thresholds[0] > min_value:
        manual_thresholds.insert(0, min_value)

    # Kullanıcı tarafından girilen eşik değerler listesinin son elemanı veri setinin maksimum değerinden küçükse, veri setinin maksimum değerini listenin sonuna ekle.
    if manual_thresholds and manual_thresholds[-1] < max_value:
        manual_thresholds.append(max_value)

    return manual_thresholds


def generate_threshold_scale(data, capacity_column):
    # Veri setindeki 'capacity_column' sütunundaki değerlere göre yüzdelik dilimleri hesapla
    percentiles = np.percentile(data[capacity_column], [20, 40, 60, 80])
    
    # Yüzdelik dilimlere göre eşik değerlerini belirle
    threshold_scale = [np.min(data[capacity_column])]
    threshold_scale.extend(percentiles)
    threshold_scale.append(np.max(data[capacity_column]))
    
    return threshold_scale


def update_map_2(user_data, capacity_column, color_palette, threshold_scale, m):
    with open('./harita_script/turkiye_haritasi.json', 'r') as f:
        turkey_map = json.load(f)


    for feature in turkey_map['features']:
        feature['properties']['name'] = normalize_text(feature['properties']['name'])

    type_flag = totaly_sure_percent(user_data, capacity_column)
    legend = capacity_column

    if type_flag:
        user_data[capacity_column] *= 100
        legend = f"{capacity_column} Yüzdesi (%)"

    if not threshold_scale:
        threshold_scale = generate_threshold_scale(user_data, capacity_column)

    folium.Choropleth(
        geo_data=turkey_map,
        data=user_data,
        columns=['Label', capacity_column],
        key_on='properties.name',
        fill_color=color_palette,
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name=legend,
        threshold_scale=adjust_threshold_scale(user_data, capacity_column, threshold_scale),
        highlight=True
    ).add_to(m)

    folium.plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
    ).add_to(m) 

    def style_function(feature):
        return {'fillColor': 'transparent', 'color': 'transparent'}

    # Creating a GeoJson layer for tooltips
    geojson_layer = folium.GeoJson(
        turkey_map,
        style_function=style_function,
        name='geojson'
    ).add_to(m)

    for feature in turkey_map['features']:
        prop = feature['properties']
        city_name = prop['name']

        if city_name in user_data['Label'].values:
            city_data = user_data[user_data['Label'] == city_name].iloc[0]
            if type_flag:
                tooltip_text = f"{city_name}: {capacity_column}: %{city_data[capacity_column]:.2f}"
            else:
                tooltip_text = f"{city_name}: {capacity_column}: {city_data[capacity_column]}"

            tooltip = folium.Tooltip(tooltip_text)
            folium.GeoJson(
                feature,
                style_function=style_function,
                tooltip=tooltip
            ).add_to(geojson_layer)


    top_branches = user_data[user_data[capacity_column] > user_data[capacity_column].mean()].sort_values(
        by=capacity_column, ascending=False)[:25]
    bottom_branches = user_data[user_data[capacity_column] < user_data[capacity_column].mean()].sort_values(
        by=capacity_column)[:25]
    

    top_branches_list = top_branches.to_dict('records')
    bottom_branches_list = bottom_branches.to_dict('records')

    m.save('static/updated_map.html')

    return m, top_branches_list, bottom_branches_list,type_flag


def update_map(user_data, capacity_column, color_palette,threshold_scale,m):
    type_flag = totaly_sure_percent(user_data, capacity_column)
    legend = capacity_column

    if type_flag:
        user_data[capacity_column] *= 100
        legend = f"{capacity_column} Yüzdesi (%)"

    if not threshold_scale:
        threshold_scale = generate_threshold_scale(user_data, capacity_column)
    filtered_features = [feature for feature in garenta_map["features"] if feature["geometry"]["type"] != "Point"]
    geo_data={"type": "FeatureCollection", "features": filtered_features}
    folium.Choropleth(
    geo_data=geo_data,
        data=user_data,
        columns=['Label', capacity_column],
        key_on='properties.Label',
        fill_color=color_palette,
        fill_opacity=0.6,
        line_opacity=0.4,
        legend_name=legend,
        threshold_scale=adjust_threshold_scale(user_data, capacity_column, threshold_scale),
    ).add_to(m)

    # Define a style function for transparency
    def style_function(feature):
        return {'fillColor': 'transparent', 'color': 'transparent'}

    # Creating a GeoJson layer for tooltips
    geojson_layer = folium.GeoJson(
        geo_data,
        style_function=style_function,
        name='geojson'
    ).add_to(m)
    folium.plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
    ).add_to(m) 
    # Adding tooltips for each feature based on user data


    marker_cluster = MarkerCluster().add_to(m)  # Marker kümesini oluştur ve haritaya ekle

    # GeoJSON dosyasındaki her bir nokta için bir döngü
    for feature in garenta_map['features']:
        if feature['geometry']["type"] == "Point":
            coordinates = feature['geometry']['coordinates']
            label = feature['properties']['Label']

            folium.Marker(
                location=[coordinates[1], coordinates[0]],  # Koordinatlar [enlem, boylam] formatında
                popup=label,  # Popup metni olarak Label kullan
                icon=folium.Icon(color='orange')  # Marker rengi turuncu
            ).add_to(marker_cluster)  # Marker kümesine ekle


    for feature in geo_data['features']:
        prop = feature['properties']
        city_name = prop['Label']
        # Check if the city is in the user data
        if city_name in user_data['Label'].values:
            city_data = user_data[user_data['Label'] == city_name].iloc[0]
            if type_flag:
                tooltip_text = f"{city_name}: {capacity_column}: %{city_data[capacity_column]:.2f}"
            else:
                tooltip_text = f"{city_name}: {capacity_column}: {city_data[capacity_column]}"

            tooltip = folium.Tooltip(tooltip_text)
            folium.GeoJson(
                feature,
                style_function=style_function,
                tooltip=tooltip
            ).add_to(geojson_layer)

    mean=user_data[capacity_column].mean()
    third=user_data[capacity_column].quantile(0.75)
    first=user_data[capacity_column].quantile(0.25)

    top_branches = user_data.loc[user_data[capacity_column] > third,("Label", capacity_column)].sort_values(
        by=capacity_column, ascending=False)
    bottom_branches = user_data.loc[user_data[capacity_column] < first,("Label", capacity_column)].sort_values(
        by=capacity_column)

    top_branches_list = top_branches.to_dict('records')
    #print(top_branches_list)
    bottom_branches_list = bottom_branches.to_dict('records')
    #print(bottom_branches_list)
    # Güncellenmiş haritayı kaydet
    m.save('static/updated_map.html')

    # DataFrame'leri değil, listeleri döndür
    return m, top_branches_list, bottom_branches_list,type_flag



@app.route('/get-sheets', methods=['POST'])
def get_sheets():
    file = request.files.get('file')
    if file:
        excel_file = BytesIO(file.read())
        xls = pd.ExcelFile(excel_file)
        sheets = xls.sheet_names
        return jsonify(sheets)
    else:
        return jsonify({"error": "No file provided"}), 400
    
@app.route('/get-columns', methods=['POST'])
def get_columns():
    file = request.files.get('file')
    sheet_name = request.form.get('sheetName')
    if file and sheet_name:
        # Dosyayı ve seçilen sheet'i okuyup sütun isimlerini al
        excel_file = BytesIO(file.read())
        if not sheet_name:  # Eğer sheet ismi verilmediyse, ilk sheet'i seç
            sheet_name = excel_file.sheet_names[0]
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        columns = df.columns.tolist()[1:]
        return jsonify(columns)
    else:
        return jsonify({"error": "No file or sheet name provided"}), 400





def new_file():
    conn_string = "DefaultEndpointsProtocol=https;AccountName=sftpdeneme;AccountKey=Ks/pBLXYECIqDTZUa9zATbahogkLAEiGFog2xc41S9YJ4Y6oiOL977t7IqKr+0+UbHfoqdIpI++4+AStX8APEw==;EndpointSuffix=core.windows.net"
    blob_service_client = BlobServiceClient.from_connection_string(conn_string)
    container_client = blob_service_client.get_container_client("sftpdemo")
    blob_client = blob_service_client.get_blob_client(container = "subegorsellestirme", blob="Koordinatlar.xlsx")
    f = open("Koordinatlar.xlsx", "wb")
    f.write(blob_client.download_blob().content_as_bytes())
    f.close()
    default_data = pd.read_excel(r''+"Koordinatlar.xlsx",sheet_name="Kapasite")

    return default_data


@app.route('/default-data')
def default_data():

    user_data=new_file()
    old_column_name = user_data.columns[0]
    user_data.rename(columns={old_column_name: "Label"}, inplace=True)
    user_data['Label'] = user_data['Label'].apply(normalize_text)
    capacity_column = 'Kapasite'
    user_data[capacity_column].fillna(0, inplace=True)
    type_flag = totaly_sure_percent(user_data, capacity_column)
    if type_flag:
        user_data[capacity_column] *= 100
    
    third=user_data[capacity_column].quantile(0.75)
    first=user_data[capacity_column].quantile(0.25)

    top_branches = user_data.loc[user_data[capacity_column] > third,("Label", capacity_column)].sort_values(
        by=capacity_column, ascending=False)
    bottom_branches = user_data.loc[user_data[capacity_column] < first,("Label", capacity_column)].sort_values(
        by=capacity_column)
    top_branches = top_branches.to_dict('records')
    bottom_branches = bottom_branches.to_dict('records')

    return jsonify({
        'topBranches': top_branches,
        'bottomBranches': bottom_branches,
        'selectedColumn': capacity_column,
        'typeFlag':type_flag
    })






# Ana sayfa ve dosya yükleme formu
@app.route('/', methods=['GET', 'POST'])
def upload_file():
   m = folium.Map(location=[37.925533, 35.06287], zoom_start=5.5,tiles="https://api.mapbox.com/styles/v1/alpdev/clt02w2jt00fx01pi171gaqsc/tiles/256/{z}/{x}/{y}@2x?access_token=pk.eyJ1IjoiYWxwZGV2IiwiYSI6ImNsc3p3dXlzejBxZDMya28ycjdjMms5cGEifQ.7rVuB4i8lTPAbidnGg6JbQ",prefer_canvas=True,attr='Mapbox Light')
   if request.method == 'POST':
        file = request.files['file'] 
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls']:
            sn = request.form.get('selected_sheet')
            print(sn)
            user_data = pd.read_excel(file,sheet_name=sn)
            old_column_name = user_data.columns[0]  
            user_data = user_data.rename(columns={old_column_name: "Label"})
            selected_column = request.form.get('selected_column')
            user_data['Label'] = user_data['Label'].apply(normalize_text)
            map_option = request.form.get('map_option')
            color_palette = request.form.get('color_palette', 'YlOrRd')
            #print(f"Seçilen Renk Paleti: {color_palette}")
            threshold_scale_input = request.form.get('threshold_scale')

            if threshold_scale_input and isinstance(threshold_scale_input, str):
                threshold_scale = list(map(int, threshold_scale_input.split(',')))
                print(threshold_scale)
            else:
                threshold_scale = []
            user_data[selected_column] = user_data[selected_column].fillna(0)


            if map_option == "0":
                m, top_branches, bottom_branches,typeflag = update_map(user_data, selected_column, color_palette, threshold_scale,m)
            elif map_option == "1":
                m, top_branches, bottom_branches,typeflag = update_map_2(user_data, selected_column, color_palette, threshold_scale,m)
            map_file_path = os.path.join(app.static_folder, 'generated_map.html')
            m.save(map_file_path)
            map_url = url_for('static', filename='generated_map.html')
            response_data = {
                'mapFile': map_url,
                'topBranches': top_branches,  # DataFrame'i list of dicts'e çevir
                'bottomBranches': bottom_branches,
                'selectedColumn': selected_column,
                'typeFlag':typeflag
            }
            return jsonify(response_data)
        
        return jsonify({})
        
   else:
        # Load default data
        default_data=new_file()
        old_column_name = default_data.columns[0]
        default_data.rename(columns={old_column_name: "Label"}, inplace=True)
        default_data['Label'] = default_data['Label'].apply(normalize_text)
        color_palette = 'YlOrRd'
        selected_column = 'Kapasite'
        default_data[selected_column].fillna(0, inplace=True)
        threshold_scale = []
        m, top_branches, bottom_branches,typeflag = update_map(default_data, selected_column, color_palette, threshold_scale, m)
        map_file_name = 'default_map.html'  
        map_file_path = os.path.join(app.static_folder, map_file_name)
        columns = default_data.columns.tolist()[1:]
        m.save(map_file_path)
        #return jsonify(response_data)
        return render_template('index.html', map_file=map_file_name, top_branches=top_branches, bottom_branches=bottom_branches, capacity_column="Kapasite", columns=columns)

   

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))  # Azure tarafından sağlanan PORT değerini kullan
    app.run(host='0.0.0.0', port=port)

