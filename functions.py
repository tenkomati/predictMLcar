import os
import time
import math
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from fastparquet import ParquetFile


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



# Function to delete CSV files older than one day
def cleanup_csv_files(directory):
    now = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.csv'):
            # Get the file's modification time
            modified_time = os.path.getmtime(filepath)
            # Calculate the time difference
            time_difference = now - modified_time
            # Define the threshold (1 day in seconds)
            threshold = 24 * 60 * 60
            if time_difference > threshold:
                os.remove(filepath)



def round_next(num):
    next_num = math.ceil(num*10)/10
    if next_num - num < 0.1:
        return round(next_num, 1)
    else:
        return round(num, 1)




def grafcantidadhist(marca,modelo,anio):

    try:
        pf=ParquetFile('./data/raw/'+marca+modelo+'historico.parq')
        df= pf.to_pandas()
    except FileNotFoundError:
        return None
    plt.clf()
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
    
    plt.clf()
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

