from flask import Flask, render_template, request
import requests
from csv import writer
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import math
import os



def round_next(num):
    next_num = math.ceil(num*10)/10
    if next_num - num < 0.1:
        return round(next_num, 1)
    else:
        return round(num, 1)


app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for car price prediction form submission
@app.route('/predict', methods=['POST'])
def pred():
    # Retrieve form data
    marca = request.form['make']
    modelo = str(request.form['model'])
    anio = request.form['year']

    naft = int(request.form["nafta"])
    km = int(request.form["kms"])
    motr = float(request.form["motor"])
    aw = int(request.form["awd"])
    manua = request.form["manual"]      


    #reemplazar espacios
    marca = marca.replace(" ","%20")
    modelo = modelo.replace(" ","%20")
    anio = anio.strip()



    url = str('https://api.mercadolibre.com/sites/MLA/search?q=' + marca + "%20" + modelo + "%20" + anio +'&category=MLA1743&id=MLA1744') 
    payload = {}
    headers = {
    'Authorization': 'Bearer APP_USR-1735483845190568-052521-4e03162aa48b46d277ba15d55d6a2357-224513779'
    }
    page = requests.request("GET", url, headers=headers, data=payload)



    #Genera JSON
    Jsonpage= page.json()



    #Extrae info
    cantidad = Jsonpage['paging']['total']
    #print('cantidad: '+str(cantidad))
    list = Jsonpage['results']


    #GENERA EL ARCHIVO CSV
    with open(str('./data/raw/'+marca+'_'+modelo+'_'+anio+'.csv'),'w',encoding='utf8',newline='') as f:
        thewriter = writer(f)
        header = ['marca','modelo','version', 'motor','nafta','precio','moneda', 'anio', 'kms','awd', 'manual','id ml', 'link']
        thewriter.writerow(header)
        cantidadrow=0
        for i in list:
                mlid = i['id']
                precio = i['price']
                moneda = i['currency_id']
                for k in i['attributes']:
                    if k['id'] == 'VEHICLE_YEAR':
                        aniox = k['value_name']
                    elif k['id'] == 'KILOMETERS':
                        kms = k['value_name']
                    elif k['id'] == 'TRIM':
                        version = k['value_name']
                    elif k['id'] == 'ENGINE_DISPLACEMENT':
                        motor = k['value_name']
                    elif k['id'] == 'FUEL_TYPE':
                        if k['value_name'] == 'Nafta':
                            nafta = 1
                        else:
                            nafta = 0
                    elif k['id'] == "TRACTION_CONTROL":
                        if k['value_name'] == 'Integral' or k['value_name'] == '4x4':
                            awd = 1
                        else:
                            awd = 0
                    elif k['id'] == "TRANSMISSION":
                        if k['value_name'] == 'Manual':
                            manual = 1
                        else:
                            manual = 0
                href = i['permalink']
                info = [marca, modelo, version, motor ,nafta, precio, moneda, aniox, kms, awd, manual, mlid, href]
                thewriter.writerow(info)
                cantidadrow += 1

        
        for x in range(1,int((cantidad/50)+1)):
            url2 = str(url + '&offset='+ str((x*50)+1))
            page = requests.request("GET", url2, headers=headers, data=payload)
            Jsonpage= page.json()
            list = Jsonpage['results']
            for i in list:
                mlid = i['id']
                precio = i['price']
                moneda = i['currency_id']
                for k in i['attributes']:
                    if k['id'] == 'VEHICLE_YEAR':
                        aniox = k['value_name']
                    elif k['id'] == 'KILOMETERS':
                        kms = k['value_name']
                    elif k['id'] == 'TRIM':
                        version = k['value_name']
                    elif k['id'] == 'ENGINE_DISPLACEMENT':
                        motor = k['value_name']
                    elif k['id'] == 'FUEL_TYPE':
                        if k['value_name'] == 'Nafta':
                            nafta = 1
                        else:
                            nafta = 0
                    elif k['id'] == "TRACTION_CONTROL":
                        if k['value_name'] == 'Integral' or k['value_name'] == '4x4':
                            awd = 1
                        else:
                            awd = 0
                    elif k['id'] == "TRANSMISSION":
                        if k['value_name'] == 'Manual':
                            manual = 1
                        else:
                            manual = 0
                href = i['permalink']
                info = [marca, modelo, version, motor,nafta, precio, moneda, aniox, kms, awd, manual, mlid, href]
                thewriter.writerow(info)
                cantidadrow += 1



    #Obtener valor del dolar del dia
    url='https://dolarhoy.com/cotizaciondolarblue'
    response = requests.get(url)
    webpage = response.content
    soup = BeautifulSoup(webpage, "html.parser")
    d = soup.find(class_="cotizacion_moneda").find(class_='value').text
    dolarstr = re.search('\d+\.\d+',d).group(0)
    print('Dolar Blue a la fecha: '+ dolarstr)
    dolarhoy = int(float(dolarstr))



    #GENERATE Dataframe
    DBauto = pd.read_csv('data/raw/'+marca+'_'+modelo+'_'+anio+'.csv')
    #cantidad = str(len(DBauto))

    #CLEAN AND TRANSFORM FEATURES

    #KMS texto a int
    #DBauto['kms'] = DBauto['kms'].str.replace(" km","").str.extract('(\d+)').astype(int)
    DBauto['kms'] = DBauto['kms'].str.extract('(\d+)').astype(int)


    #PRECIO dolares a pesos
    for i in range(len(DBauto)):
        x = int(DBauto.loc[i,'precio'])
        if DBauto.loc[i,'moneda'] == 'USD':
            DBauto.loc[i,'precio'] = x*dolarhoy
            DBauto.loc[i,'moneda'] = 'ARS'


    '''
    #VERSION
    DBauto['version'] = DBauto['version'].str.lower()

    #busca si existe el volumen del  motor y genera una nueva columna con ese valor para constatar vs motor
    pattern = re.compile(r'\d+\.\d+') # This regular expression matches a string containing a number followed by a dot and another number.
    for i in range(len(DBauto)):
        version = str(DBauto.iloc[i]["version"]) # Convert the version to a string just in case.
        if pattern.findall(version): # Check if the pattern exists in the version string.
            DBauto.loc[i,'v'] = float(pattern.findall(version)[0])
        else:
            DBauto.loc[i,'v'] = 0
    '''


    #MOTOR
    DBauto['motor'] = DBauto['motor'].str.extract('(\d+(?:\.\d+)?)').astype(float)
    for i in range(len(DBauto)):
        if DBauto.loc[i,'motor'] > 12.0: #looking for engines in cc and convert to litres
            DBauto.loc[i,'motor'] = (DBauto.loc[i,'motor'] / 1000)
            DBauto.loc[i,'motor'] = round_next(DBauto.loc[i,'motor']) #round for the next float with one decimal

    #FEATURE ENGINEER
    #precio x km
    DBauto['precioxkm'] = DBauto['precio']/DBauto['kms']
    DBauto = DBauto.sort_values(by=['precioxkm'],ascending=True)


    #STATS
    #calculo cuantiles y iQR de PRECIO
    q1 = np.quantile(DBauto['precio'],0.25)
    q3 = np.quantile(DBauto['precio'],0.75)
    iqr = q3 - q1

    #cutoff outliers de precio
    lower = q1-1.5*iqr
    upper = q3+1.5*iqr


    DBautosinoutliers = DBauto[(DBauto['precio'] >= lower) & (DBauto['precio'] <= upper)]

    #calculo cuantiles y iQR de PRECIO X KM
    q11 = np.quantile(DBautosinoutliers['precioxkm'],0.25)
    q33 = np.quantile(DBautosinoutliers['precioxkm'],0.75)
    iqr1 = q33 - q11

    #cutoff outliers de PRECIO X KM
    lower1 = q11-iqr1

    #Autos baratos x precio
    DBautosbaratos = DBauto[(DBauto['precio'] < lower)]


    #Autos baratos x precioxkm
    DBbuenprecioxkm = DBautosinoutliers[(DBautosinoutliers['precioxkm'] <= lower1)].sort_values(by=['precio'],ascending=True)


    #PRINTS
    
    print('Estadisticas: \n')
    precio_mean = DBautosinoutliers['precio'].mean()
    precio_med = DBautosinoutliers['precio'].median()
    kms_mean = DBautosinoutliers['kms'].mean()
    kms_med = DBautosinoutliers['kms'].median()



    versionFREQ = DBautosinoutliers['version'].value_counts().rename_axis('version').reset_index(name='cantidad')
    primero = versionFREQ.iloc[0,0]
    segundo = versionFREQ.iloc[1,0]
    tercero = versionFREQ.iloc[2,0]

    #GENERA UN DB NUEVO PARA MODELAR
    Processedx = DBautosinoutliers[['motor','nafta','precio','kms','awd','manual']]
    #DBmodel.to_csv('./data/processed/DBmodel.csv',index=False)

    X = Processedx.drop('precio',axis=1)
    y = Processedx['precio']



    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    # Define your pipeline
    pipeline1 = Pipeline([
        ('scaler', StandardScaler()),  # Optional: Scale features
        ('regressor', DecisionTreeRegressor())  # Decision tree regressor
    ])

    pipeline2 = Pipeline([
        ('scaler', StandardScaler()),  # Optional: Scale features
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  
    ])
    # Train your pipeline
    pipeline1.fit(X_train, y_train)

    pipeline2.fit(X_train, y_train)

    # Make predictions
    y_pred1 = pipeline1.predict(X_test)
    y_pred2 = pipeline2.predict(X_test)

    # Calculate metrics
    rmse1 = mean_squared_error(y_test, y_pred1, squared=False)
    print("Tree Regressor RMSE: ", rmse1)
    r2_score1 = r2_score(y_test, y_pred1)
    print("Tree Regressor R2: ",r2_score1)


    rmse2 = mean_squared_error(y_test, y_pred2, squared=False)
    print("Random Forest RMSE:", rmse2)
    r2_score2 = r2_score(y_test, y_pred2)
    print("Random Forest R2: ",r2_score2)

    #def predecir(motor,nafta,kms,awd,manual):
            # Create DataFrame
    data = {
        'motor': [motr],
        'nafta': [naft],
        'kms': [km],
        'awd': [aw],
        'manual': [manua]
    }
    df = pd.DataFrame(data)

    #Si no se ingresaron datos para la prediccion se utiliza el promedio de kms y los valores de DBautosinoutliers
    if df.loc[0,'awd'] == None:
        df.loc[0,'awd'] = DBautosinoutliers['awd'].value_counts().index[0]
    if df.loc[0,'kms'] == None:
        df.loc[0,'kms'] = DBautosinoutliers['kms'].mean()
    if df.loc[0,'manual'] == None:
        df.loc[0,'manual'] = DBautosinoutliers['manual'].value_counts().index[0]
    if df.loc[0,'motor'] == None:
        df.loc[0,'motor'] = DBautosinoutliers['motor'].value_counts().index[0]
    if df.loc[0,'nafta'] == None:
        df.loc[0,'nafta'] = DBautosinoutliers['nafta'].value_counts().index[0]

    

    ypred1 = pipeline1.predict(df)
    ypred2 = pipeline2.predict(df)
    print('precio predecido con treee model: ', ypred1)
    print('precio predecido con rforest: ', ypred2)

    # Calcular los cuatro valores obtenidos por los modelos predictivos
    valor_1 = int(ypred1[0])
    valor_2 = int(ypred2[0])
    #valor_3 = int(ypred3[0])
    #valor_4 = int(ypred4[0])


    # Crear el gráfico usando Seaborn
    sns.set(style="whitegrid")
    ax = sns.histplot(data=DBauto,x='precio')
    ax.axvline(precio_mean, color='red', linestyle='--', label='Promedio')
    ax.axvline(precio_med, color = 'orange', linestyle= '--', label='Mediana' )
    ax.axvline(lower, color='orange', linestyle='--', label='Límite Inferior')
    ax.axvline(upper, color='orange', linestyle='--', label='Límite Superior')
    ax.axvline(valor_1, color='green', linestyle='-', label='TreeRegressor')
    ax.axvline(valor_2, color='blue', linestyle='-', label='RandomForest')
    #ax.axvline(valor_3, color='purple', linestyle='-', label='KNN')
    #ax.axvline(valor_4, color='brown', linestyle='-', label='LinearRegression')
    plt.tight_layout()
    # Añadir leyenda
    ax.legend(loc='upper left',bbox_to_anchor=(1, 1.1))

    ax.set(xlim=(lower*0.95, upper*1.05))
    plt.ylabel('Cantidad')
    plt.title(marca+" "+modelo+" "+str(anio)+" en ML")

    # Mostrar el gráfico
    sns.despine()
    temp_file = 'static/temp_file.png'
    plt.savefig(temp_file, bbox_inches='tight')
    plt.close()


    # Render the prediction result template with the predicted price
    return render_template('result2.html',
                cantidad=cantidad,
                dolarhoy=dolarhoy,
                marca=marca,
                modelo=modelo,
                anio=anio,
                lower=lower,
                upper=upper,
                precio_mean=precio_mean,
                precio_med=precio_med,
                kms_mean=kms_mean,
                kms_med=kms_med,
                ypred1=ypred1[0],
                ypred2=ypred2[0],
                DBautosbaratos=DBautosbaratos,
                DBbuenprecioxkm=DBbuenprecioxkm,
                rmse1=rmse1,
                r2_score1=r2_score1,
                rmse2=rmse2,
                r2_score2=r2_score2,
                plot_path=temp_file
                )


if __name__ == '__main__':
    #app.run(debug=True, )
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



        












