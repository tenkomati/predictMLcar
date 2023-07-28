import os
import time
import math
import csv
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from fastparquet import ParquetFile
from tqdm import tqdm
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import re
import json

#FUNCIONES PRINCIPALES
def fetch_data_from_mercadolibre(marca, modelo, anio, access_token):
    url = f'https://api.mercadolibre.com/sites/MLA/search?q={marca}%20{modelo}%20{anio}&category=MLA1743&id=MLA1744'
    payload = {}
    headers = {
        'Authorization': f'Bearer {access_token}'
    }

    with requests.Session() as session:
        response = session.get(url, headers=headers, data=payload)
        response.raise_for_status()  # Raise an exception if the request was unsuccessful

    return response.json()

def extract_attributes(attributes):
    attribute_mapping = {
        'VEHICLE_YEAR': 'aniox',
        'MODEL': 'model',
        'KILOMETERS': 'kms',
        'TRIM': 'version',
        'ENGINE_DISPLACEMENT': 'motor',
        'FUEL_TYPE': 'nafta',
        'TRACTION_CONTROL': 'awd',
        'TRANSMISSION': 'manual',
    }

    extracted_attributes = {key: None for key in attribute_mapping.values()}

    for attribute in attributes:
        attribute_id = attribute['id']
        attribute_value = attribute['value_name']

        if attribute_id in attribute_mapping:
            extracted_attributes[attribute_mapping[attribute_id]] = attribute_value.lower() if attribute_id == 'MODEL' else attribute_value

    return extracted_attributes

def DBauto_funcion(marca, modelo, anio, access_token):
    try:
        api_response = fetch_data_from_mercadolibre(marca, modelo, anio, access_token)
    except requests.exceptions.RequestException as e:
        raise Exception(f'Error fetching data from MercadoLibre API: {e}')

    total_results = api_response['paging']['total']
    if total_results >= 1000:
        total_results = 999

    listings = api_response['results']

    data = []
    for listing in listings:
        mlid = listing['id']
        precio = listing['price']
        moneda = listing['currency_id']
        attributes = listing['attributes']
        extracted_attributes = extract_attributes(attributes)

        href = listing['permalink']
        info = [marca, extracted_attributes['model'], extracted_attributes['version'], extracted_attributes['motor'],
                extracted_attributes['nafta'], precio, moneda, extracted_attributes['aniox'], extracted_attributes['kms'],
                extracted_attributes['awd'], extracted_attributes['manual'], mlid, href]
        data.append(info)

    if total_results > 50:
        for x in range(1, int((total_results / 50) + 1)):
            try:
                api_response = fetch_data_from_mercadolibre(marca, modelo, anio, access_token)
            except requests.exceptions.RequestException as e:
                print(f'Error fetching data from MercadoLibre API: {e}')
                continue

            listings = api_response['results']
            for listing in listings:
                precio = listing['price']
                moneda = listing['currency_id']
                attributes = listing['attributes']
                extracted_attributes = extract_attributes(attributes)

                href = listing['permalink']
                info = [marca, extracted_attributes['model'], extracted_attributes['version'], extracted_attributes['motor'],
                        extracted_attributes['nafta'], precio, moneda, extracted_attributes['aniox'], extracted_attributes['kms'],
                        extracted_attributes['awd'], extracted_attributes['manual'], mlid, href]
                data.append(info)

    columns=['marca','modelo','version', 'motor','nafta','precio','moneda', 'año', 'kms','awd', 'manual','id ml', 'link']
    DBauto = pd.DataFrame(data, columns=columns)
    return DBauto

def modelado(X_train, X_test, y_train, y_test,modelo,DatosPredecir):

    if modelo == 'tree':
        MODELO = DecisionTreeRegressor()
    elif modelo == 'rf':
        MODELO = RandomForestRegressor(n_estimators=100, random_state=42)
    elif modelo == 'knn':
        MODELO = KNeighborsRegressor(n_neighbors=5)
    elif modelo == 'linear':
        MODELO = LinearRegression()

    #CREAMOS LAS PIPELINES PARA CADA MODELO
    pipeline = Pipeline([('scaler', StandardScaler()),('regressor', MODELO)])

    #TRAIN
    pipeline.fit(X_train, y_train)

    #PREDICT
    y_pred = pipeline.predict(X_test)

    #METRICS
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    #PREDICCION
    valor = int(pipeline.predict(DatosPredecir)[0])

    return valor,rmse,r2


#GRAFICOS

def grafcantidadhist(marca,modelo,anio):

    try:
        pf=ParquetFile('./data/raw/'+marca+modelo+'historico.parq')
        df= pf.to_pandas()
    except FileNotFoundError:
        return None
    
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x='año', y='cantidad')
    ax.set_title('Cantidad x año')
    ax.set_xticks(ax.get_xticks())
    ax.tick_params(axis='x', rotation=90)
    highlight_year= df.loc[df['año']==int(anio)].index[0]
    ax.patches[highlight_year].set_facecolor('red')
    ax.get_xticklabels()[highlight_year].set_color('red')
    temp_file = 'static/grafcantidadhist.png'
    plt.savefig(temp_file, bbox_inches='tight')
    plt.close()
    return temp_file

def grafpreciohist(marca,modelo,anio):
    try:
        pf=ParquetFile('./data/raw/'+marca+modelo+'historico.parq')
        df= pf.to_pandas()
    except FileNotFoundError:
        return None
    
    
    sns.set_style("whitegrid")
    ax = sns.lineplot(data=df, x='año', y='precio')
    ax.set_title('Precio promedio x año')
    ax.set_xticks(ax.get_xticks())
    ax.tick_params(axis='x', rotation=90)
    ax.axvline(anio, color = 'orange', linestyle= '-', label='año' )
    precio = df.loc[df['año']== anio,'precio'].values[0]
    plt.scatter(anio, precio, color='red', marker='o', s=100)
    plt.annotate(f'{precio:.0f}', (anio, precio), textcoords="offset points", xytext=(-10,10), ha='right')
    temp_file = 'static/grafpreciohist.png'
    plt.savefig(temp_file, bbox_inches='tight')
    plt.close()
    return temp_file


def grafvalorcantidad(db,mean,prediccion,marca,modelo,anio):
    sns.set_style("whitegrid") #seteo el estilo de grafico aqui, ya que una parte está dentro del if
    ax = sns.histplot(data=db,x='precio')
    ax.axvline(mean, color='black', linestyle='--', label='Promedio')
    ax.axvline(prediccion, color='green', linestyle='-', label='Prediccion')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    ax.legend(loc='upper left',bbox_to_anchor=(1, 1.1))
    plt.ylabel('Cantidad')
    plt.xlabel('Precio(en USD)')
    plt.title(marca.capitalize()+" - "+modelo.capitalize()+" ( "+str(anio)+" )")
    sns.despine()
    temp_file = 'static/temp_file.png'
    plt.savefig(temp_file, bbox_inches='tight')
    plt.close()
    return temp_file

#FUNCIONES PARA CARGAR MARCAS Y MODELOS
def load_car_models():
    car_models = {}
    with open('./data/modelosmarcas.csv', 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            marca = row[3]
            modelo = row[1]
            if marca in car_models:
                car_models[marca].append(modelo)
            else:
                car_models[marca] = [modelo]
    car_models.pop('marca')
    return car_models


def get_marcas():
    marcas = []
    with open('./data/marcas.csv', 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            marca = row[1]
            if marca not in marcas:
                marcas.append(marca)
        marcas[0] = 'ingrese un valor'
    return marcas

def get_trims():
    #genera un diccionario con los trims y los id de cada modelo de auto. No es necesario llamar a esta funcion otra vez ya que ya se realizó. A menos que se desee obtener tambien la cantidad de cada trim publicado
    id_trims = {}
    
    url = 'https://api.mercadolibre.com/catalog_domains/MLA-CARS_AND_VANS/attributes/TRIM/top_values'
    with open('./data/modelosmarcas.csv', 'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for modelo in tqdm(reader,desc='modelos', unit='modelo'):
            trims = [] #reinicio los trims para un modelo nuevo
            id= modelo[0] #id del modelo
            #API REQUEST
            
            payload = json.dumps({
                "known_attributes": [
                    {
                        "id": "MODEL",
                        "value_id": id
                    }
                ]
            })
            headers = {
                'Content-Type': 'application/json'
                    }
            page = requests.request("POST", url, headers=headers, data=payload)
            Jsonpage= page.json()
            for trim in tqdm(Jsonpage,desc='trims'):
                name = trim['name']  #el nombre de cada trim
                trims.append(name)  #apendo a lista de trims
            id_trims[id] = trims  #finalizo el bucle para el modelo y añado al dic id de modelo + trims
    pdtrims = pd.DataFrame(columns=['model_ID','trim'])

    for model_id, trimss in id_trims.items():
        for trim in trimss:
            pdtrims = pdtrims.append({'model_ID': model_id, 'trim': trim}, ignore_index=True)


#FUNC AUX
def round_next(num):
    next_num = math.ceil(num*10)/10
    if next_num - num < 0.1:
        return round(next_num, 1)
    else:
        return round(num, 1)
    
def get_dolarhoy():
    #Obtener valor del dolar del dia
    url='https://dolarhoy.com/cotizaciondolarblue'
    response = requests.get(url)
    webpage = response.content
    soup = BeautifulSoup(webpage, "html.parser")
    d = soup.find(class_="cotizacion_moneda").find(class_='value').text
    dolarstr = re.search('\d+\.\d+',d).group(0)
    dolarhoy = int(float(dolarstr))
    return dolarhoy