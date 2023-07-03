import os
import time
from datetime import datetime, timedelta
import math
import csv


def load_car_models():
    car_models = {}
    with open('./data/modelosmarcas.csv', 'r') as csvfile:
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
    with open('./data/marcas.csv', 'r') as csvfile:
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

