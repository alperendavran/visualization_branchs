from flask import Flask, render_template, request, session, jsonify
import pandas as pd
import folium
import json
import os
from tempfile import mkdtemp
import numpy as np
import sys
import unicodedata
import unidecode
import branca.colormap as cm
import folium.plugins as plugins
import tempfile
import uuid  # Eklenen kütüphane


app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Dosyaların yükleneceği klasör

# Gerekli klasör yoksa oluştur
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def adjust_threshold_scale(data, column, manual_thresholds):
    min_value = data[column].min()
    max_value = data[column].max()

    # Kullanıcı tarafından girilen eşik değerler listesinin ilk elemanı veri setinin minimum değerinden büyükse,
    # veri setinin minimum değerini listenin başına ekle.
    if manual_thresholds and manual_thresholds[0] > min_value:
        manual_thresholds.insert(0, min_value)

    # Kullanıcı tarafından girilen eşik değerler listesinin son elemanı veri setinin maksimum değerinden küçükse,
    # veri setinin maksimum değerini listenin sonuna ekle.
    if manual_thresholds and manual_thresholds[-1] < max_value:
        manual_thresholds.append(max_value)

    return manual_thresholds

def normalize_text(text):
    text = unidecode.unidecode(text)
    text = text.lower()
    text = text.strip()
    text = " ".join(text.split())
    # Gerekiyorsa başka temizlik işlemleri burada yapılabilir
    return text.title()


# GeoJSON verisini yükle
with open('map-2.geojson', 'r') as f:
    turkey_map = json.load(f)


with open('turkey_yeni.geojson', 'r') as f:
    raw_map = json.load(f)

garenta=pd.read_excel("garenta_branches.xlsx")
garenta['City'] = garenta['City'].apply(normalize_text)

for feature in turkey_map['features']:
    if 'Label' not in feature['properties'] or feature['properties']['Label'] is None:
      feature['properties']['Label'] = 'Placeholder'
    feature['properties']['Label'] = normalize_text(feature['properties']['Label'])

for feature in raw_map['features']:
    feature['properties']['name'] = normalize_text(feature['properties']['name'])

def generate_threshold_scale(data, capacity_column):
    # Veri setindeki 'capacity_column' sütunundaki değerlere göre yüzdelik dilimleri hesapla
    percentiles = np.percentile(data[capacity_column], [20, 40, 60, 80])
    
    # Yüzdelik dilimlere göre eşik değerlerini belirle
    threshold_scale = [np.min(data[capacity_column])]
    threshold_scale.extend(percentiles)
    threshold_scale.append(np.max(data[capacity_column]))
    
    return threshold_scale


def create_initial_map():
    m = folium.Map(location=[37.925533, 35.06287], zoom_start=5.5,tiles="cartodbpositron")

    folium.plugins.Fullscreen(
    position="topright",
    title="Expand me",
    title_cancel="Exit me",
    force_separate_button=True,
    ).add_to(m) 

    def style_function(feature):
        if feature['properties']['name'] in garenta["City"].to_list():
            return {
                'fillColor': '#FF8C00',
                'color': '#FF8C00',
                'weight': 1,
                'fillOpacity': 0.3,
                'line_opacity': 0.01
               }
        else:
            return {
                'fillColor': 'black',
                'color': 'transparent',
                'fillOpacity': 0.2,
                'line_opacity': 0.4
            }

    folium.GeoJson(raw_map, style_function=style_function).add_to(m)

    m.save('static/initial_map.html')

    return m




for feature in turkey_map['features']:
    if 'Label' not in feature['properties'] or feature['properties']['Label'] is None:
      feature['properties']['Label'] = 'Placeholder'
    feature['properties']['Label'] = normalize_text(feature['properties']['Label'])

def totaly_sure_percent(df, capacity_column):
    for i, row in df.iterrows():
        if row[capacity_column]!="":
            if row[capacity_column] > 1:
                return False
    return True


def update_map_2(user_data, capacity_column, color_palette,threshold_scale):
    type_flag = totaly_sure_percent(user_data, capacity_column)
    legend = capacity_column

    if type_flag:
        user_data[capacity_column] = user_data[capacity_column].apply(lambda x: x * 100 if 0 < x < 1 else x)
        legend = capacity_column + " Yüzdesi (%)"
    #m = folium.Map(location=[37.925533, 35.06287], zoom_start=5.5,tiles="cartodbpositron")
    m = folium.Map(location=[39.925533, 32.866287], zoom_start=5,tiles="openstreetmap")

    if threshold_scale==[]:
        threshold_scale = generate_threshold_scale(user_data, capacity_column)

    folium.Choropleth
    folium.Choropleth(
        geo_data=raw_map,
        data=user_data,
        columns=['Label', capacity_column],
        key_on='properties.name',
        fill_color=color_palette,
        fill_opacity=0.7,
        line_opacity=0.4,
        legend_name=legend,
        threshold_scale=threshold_scale,
        highlight = True

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
        raw_map,
        style_function=style_function,
        name='geojson'
    ).add_to(m)

    # Adding tooltips for each feature based on user data
    for feature in raw_map['features']:
        prop = feature['properties']
        city_name = prop['name']
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


    top_branches = user_data[user_data[capacity_column] > user_data[capacity_column].mean()].sort_values(
        by=capacity_column, ascending=False)[:10]
    bottom_branches = user_data[user_data[capacity_column] < user_data[capacity_column].mean()].sort_values(
        by=capacity_column)[:10]

    top_branches_list = top_branches.to_dict('records')
    bottom_branches_list = bottom_branches.to_dict('records')

    m.save('static/updated_map.html')

    return m, top_branches_list, bottom_branches_list


def update_map(user_data, capacity_column, color_palette,threshold_scale):
    type_flag = totaly_sure_percent(user_data, capacity_column)
    legend = capacity_column

    if type_flag:
        user_data[capacity_column] = user_data[capacity_column].apply(lambda x: x * 100 if 0 < x < 1 else x)
        legend = capacity_column + " Yüzdesi (%)"
    m = folium.Map(location=[37.925533, 35.06287], zoom_start=5.5,tiles="cartodbpositron")
    #m = folium.Map(location=[37.925533, 35.06287], zoom_start=5.5,tiles="openstreetmap")

    if threshold_scale:
        threshold_scale = adjust_threshold_scale(user_data, capacity_column, threshold_scale)
        print(threshold_scale)

    else:
        threshold_scale = generate_threshold_scale(user_data, capacity_column)
        print("zoert")

    folium.Choropleth(
        geo_data=turkey_map,
        data=user_data,
        columns=['Label', capacity_column],
        key_on='properties.Label',
        fill_color=color_palette,
        fill_opacity=0.6,
        line_opacity=0.4,
        legend_name=legend,
        threshold_scale=threshold_scale,

    ).add_to(m)

    # Define a style function for transparency
    def style_function(feature):
        return {'fillColor': 'transparent', 'color': 'transparent'}

    # Creating a GeoJson layer for tooltips
    geojson_layer = folium.GeoJson(
        turkey_map,
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
    for feature in turkey_map['features']:
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

    top_branches = user_data[user_data[capacity_column] > user_data[capacity_column].mean()].sort_values(
        by=capacity_column, ascending=False)[:10]
    bottom_branches = user_data[user_data[capacity_column] < user_data[capacity_column].mean()].sort_values(
        by=capacity_column)[:10]

    top_branches_list = top_branches.to_dict('records')
    bottom_branches_list = bottom_branches.to_dict('records')

    # Güncellenmiş haritayı kaydet
    m.save('static/updated_map.html')

    # DataFrame'leri değil, listeleri döndür
    return m, top_branches_list, bottom_branches_list

@app.route('/get-columns', methods=['POST'])
def get_columns():
    file = request.files.get('file')
    if not file:
        return jsonify({'error': 'Dosya bulunamadı'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Dosyayı oku ve sütun isimlerini al
    df = pd.read_excel(filepath)
    columns = df.columns.tolist()[1:]
    return jsonify(columns)


# Ana sayfa ve dosya yükleme formu
@app.route('/', methods=['GET', 'POST'])
def upload_file():
   
   if request.method == 'POST':
        file = request.files['file'] 
        if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ['xlsx', 'xls']:            
            user_data = pd.read_excel(file)
            old_column_name = request.form.get('GarentaŞubeInput')  # Assuming this gets the old column name correctly
            user_data = user_data.rename(columns={old_column_name: "Label"})
            selected_column = request.form.get('selected_column')
            user_data['Label'] = user_data['Label'].apply(normalize_text)
            map_option = request.form.get('map_option')
            color_palette = request.form.get('color_palette', 'YlOrRd')
            print(f"Seçilen Renk Paleti: {color_palette}")
            threshold_scale_input = request.form.get('threshold_scale')

            if threshold_scale_input and isinstance(threshold_scale_input, str):
                threshold_scale = list(map(int, threshold_scale_input.split(',')))
                print(threshold_scale)
            else:
                threshold_scale = []
            user_data[selected_column] = user_data[selected_column].fillna(0)
            if user_data[['Label', selected_column]].isnull().any().any():
                return jsonify({"error": "Critical columns contain null values"}), 400

            if 'Label' not in user_data.columns or user_data['Label'].isnull().sum() > 0:
                return jsonify({"error": "DataFrame 'Label' column issue detected. You must add your data under the 'Label' column"}), 400

            if map_option == "0":
                m, top_branches, bottom_branches = update_map(user_data, selected_column, color_palette, threshold_scale)
            elif map_option == "1":
                m, top_branches, bottom_branches = update_map_2(user_data, selected_column, color_palette, threshold_scale)

            return render_template('index.html', map_file='updated_map.html', top_branches=top_branches,
                                bottom_branches=bottom_branches, capacity_column=selected_column,threshold_scale=threshold_scale)
        else:
            return jsonify({'error': 'Dosya yüklenmedi ve önceki yükleme bulunamadı.'}), 400
        
   else:
        create_initial_map()
        return render_template('index.html', map_file='initial_map.html', top_branches=[], bottom_branches=[],
                               capacity_column="", columns=[])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))  # Azure tarafından sağlanan PORT değerini kullan
    app.run(host='0.0.0.0', port=port)
