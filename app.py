from flask import Flask, render_template, request, jsonify
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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from functions import cleanup_csv_files, round_next, load_car_models, get_marcas, grafpreciohist,grafcantidadhist
import os
from thefuzz import fuzz


# OBTENIENDO marcas y modelos (desde una funcion que toma un csv y un txt donde estan estos valores)
car_models = load_car_models()
car_brands = get_marcas()


app = Flask(__name__)

# Define the home route
@app.route('/')
def home():
    return render_template('index.html',                    
                car_brands=car_brands)

#Ruta para obtener los modelos segun la marca seleccionada
@app.route('/get_car_models', methods=['POST'])
def get_car_models():
    selected_brand = request.form['brand']
    models = car_models.get(selected_brand, [])
    return jsonify({'car_models': models})

# Ruta para la prediccion
@app.route('/predict', methods=['POST'])
def pred():
    # INPUTS REQUERIDOS
    marca = str(request.form['brand']).lower().strip()
    modelo = str(request.form['model']).lower().strip()
    anio = request.form['year']

    # INPUTS NO REQUERIDOS
    #versions = str(request.form['version']).lower().strip() añade la opcion de elegir version tipo XLS XR RS etc, pero requiere mas desarrollo

    motr = request.form["motor"]
    try:
        motr = float(motr)
    except ValueError:
        motr = None

    naft = request.form["nafta"]
    try:
        naft= int(naft)
    except ValueError:
        naft = None

    km = request.form["kms"]
    try:
        km = int(km)
    except ValueError:
        km = None
        
    aw = request.form["awd"]
    try:
        aw = int(aw)
    except ValueError:
        aw = None

    manua = request.form["manual"]
    try:
        manua = int(manua)
    except ValueError:
        manua = None


  


# Crea dataframe con valores para predecir con los mismos campos que utilizan los modelos 
    data = {
        'motor': [motr],
        'nafta': [naft],
        'kms': [km],
        'awd': [aw],
        'manual': [manua]
    }
    DatosPredecir = pd.DataFrame(data)

#API REQUEST
    url = str('https://api.mercadolibre.com/sites/MLA/search?q=' + marca + "%20" + modelo + "%20" + (str(anio)).strip() +'&category=MLA1743&id=MLA1744').replace(' ','%20')
    payload = {}
    headers = {
    'Authorization': 'Bearer APP_USR-1735483845190568-052521-4e03162aa48b46d277ba15d55d6a2357-224513779'
    }
    page = requests.request("GET", url, headers=headers, data=payload)

    #Genera JSON
    Jsonpage= page.json()

    #Extrae info
    cantidad = Jsonpage['paging']['total']
    if cantidad >= 1000: #la restriccion de Mercadolibre para las busquedas es hasta 1000 resultados, y sino el codigo genera error
            cantidad = 999
         
    list = Jsonpage['results']


    #GENERA EL ARCHIVO CSV
    with open(str('./data/raw/'+marca+'_'+modelo+'_'+anio+'.csv'),'w',encoding='utf8',newline='') as f:
        thewriter = writer(f)
        header = ['marca','modelo','version', 'motor','nafta','precio','moneda', 'año', 'kms','awd', 'manual','id ml', 'link']
        thewriter.writerow(header)
        cantidadrow=0
        for i in list:
                mlid = i['id']
                precio = i['price']
                moneda = i['currency_id']
                for k in i['attributes']:
                    if k['id'] == 'VEHICLE_YEAR':
                        aniox = k['value_name']
                    elif k['id'] == 'MODEL':
                        model = str(k['value_name']).lower()
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
                info = [marca, model, version, motor ,nafta, precio, moneda, aniox, kms, awd, manual, mlid, href]
                thewriter.writerow(info)
                cantidadrow += 1

        #for loop para el resto de las paginas utilizando el parametro offset
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
                    elif k['id'] == 'MODEL':
                        model = str(k['value_name']).lower()
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
                info = [marca, model, version, motor ,nafta, precio, moneda, aniox, kms, awd, manual, mlid, href]
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
    DBauto = pd.read_csv('./data/raw/'+marca+'_'+modelo+'_'+anio+'.csv')
    

    #LImpieza y transformación de features (columnas)
    #Estos primeros pasos son para eliminar modelos y años que no sean los solicitados, ya que mercadolibre a veces incluye otros resultados.
    
    #elimina otros años
    DBauto = DBauto[DBauto['año'] == int(anio)]

    #crea col para comparar modelo solicitado vs modelos en la busqueda
    DBauto['m%'] = DBauto['modelo'].apply(lambda x: fuzz.ratio(str(x),str(modelo)))

    #elimina otros modelos
    DBauto = DBauto[DBauto['m%'] > 80]

    #actualiza cantidad de resultados reales del modelo y año seleccionado
    cantidad = len(DBauto)
    
    #SI LA BUSQUEDA ARROJA RESULTADOS ENTONCES SE EJECUTA el CODIGO, sino te avisa que no hay y volves a empezar
    if len(DBauto) > 0:
        DBauto.reset_index(inplace=True) #reset index porque generaba problemas al buscar por filas inexistentes

        #KMS texto a int
        DBauto['kms'] = DBauto['kms'].str.extract('(\d+)').astype(int)


        #PRECIO dolares a pesos utilizando la cotizacion dolarhoy blue
        for i,row in DBauto.iterrows():
                x = int(row['precio'])
                if row['moneda'] == 'ARS':
                        DBauto.at[i,'precio'] = x/dolarhoy
                        DBauto.at[i,'moneda'] = 'USD'


        #MOTOR
        #se encontraron muchos modelos que estaban cargados como centimetros cubicos con valores extraños como 1206cc por lo que tuve que armar una funcion que normalice los datos
        DBauto['motor'] = DBauto['motor'].str.extract('(\d+(?:\.\d+)?)').astype(float)
        for i,row in DBauto.iterrows():
                if row['motor'] > 12.0:         #busca motores en cc y conviertes a litros
                        DBauto.at[i,'motor'] = round_next(DBauto.at[i,'motor'] / 1000)  #redondea al proximo decimal y si tiene unidades en el segundo a tercer decimal lo redondea para arriba

        #FEATURE ENGINEER (columnas desarrolladas)
        #precio x km
        DBauto['precioxkm'] = DBauto['precio']/DBauto['kms']
        
        #STATS
        #Se utiliza cuantiles y IQR para analizar los precios y eliminar todos los valores atipicos.

        #Calculo cuantiles y iQR de PRECIO
        q1 = np.quantile(DBauto['precio'],0.25)
        q3 = np.quantile(DBauto['precio'],0.75)
        iqr = q3 - q1

        #Cutoff outliers de precio: Son los puntos donde los valores ya se determinan como atipicos.
        if (q1-1.5*iqr) < 0:
                lower = 0
        else:
                lower = q1-1.5*iqr
        upper = q3+1.5*iqr

        #DB solo de los autos entre precios normales de lower y upper
        DBautosinoutliers = DBauto[(DBauto['precio'] >= lower) & (DBauto['precio'] <= upper)].copy(deep=True)

        #Autos baratos por precio
        DBautosbaratos = DBauto[(DBauto['precio'] < lower)].copy(deep=True)
        DBautosbaratos = DBautosbaratos[['version','motor','precio','kms','precioxkm','link']]

        #calculo cuantiles y iQR de PRECIO X KM
        q11 = np.quantile(DBauto['precioxkm'],0.25)
        q33 = np.quantile(DBauto['precioxkm'],0.75)
        iqr1 = q33 - q11

        #cutoff outliers de PRECIO X KM
        if q11-iqr1 < 0:
            lower1= q11
        else:
            lower1 = q11-iqr1

        #Autos baratos x precioxkm
        DBbuenprecioxkm = DBauto[(DBauto['precioxkm'] <= lower1)].copy(deep=True).sort_values(by=['precioxkm'],ascending=True)
        DBbuenprecioxkm = DBbuenprecioxkm[['version','motor','precio','kms','precioxkm','link']]

        precio_mean = DBautosinoutliers['precio'].mean()
        precio_med = DBautosinoutliers['precio'].median()
        kms_mean = DBautosinoutliers['kms'].mean()
        kms_med = DBautosinoutliers['kms'].median()

        #PRINTS TERMINAL
        print("\n")
        print('Analisis de todos los '+ marca +" "+ modelo +" "+str(anio)+" publicadas en ML \n")
        print("\n")
        print('unidades publicadas: '+str(len(DBautosinoutliers)))
        print("\n")

        #Versiones mas frecuentes segun la busqueda
        versionFREQ = DBautosinoutliers['version'].value_counts().rename_axis('Version').reset_index(name='Cantidad')
        print('Las versiones mas comunes son')
        versionFREQ = versionFREQ.iloc[0:5,:] .copy(deep=True)
        print(versionFREQ.head(5))

        print("Precio promedio {:,.0f}".format(precio_mean))
        print("Precio mediano {:,.0f}".format(precio_med))
        print("Kilometros promedio {:,.0f}".format(kms_mean))
        print("Kilometros mediano {:,.0f}".format(kms_med))

        tamanioDB = len(DBautosinoutliers)
        sns.set_style("whitegrid") #seteo el estilo de grafico aqui, ya que una parte está dentro del if
        sns.color_palette("flare", as_cmap=True)

        if tamanioDB > 10: #SI HAY MAS DE 10 RESULTADOS EN LA BUSQUEDA SE GENERAN LAS PREDICCIONES
                    
                #GENERA UN DB NUEVO PARA MODELAR
                Processedx = DBautosinoutliers[['motor','nafta','precio','kms','awd','manual']].copy(deep=True)

                X = Processedx.drop('precio',axis=1)
                y = Processedx['precio']


                # DIVIDIMOS LOS DATOS EN DOS SETS TRAIN Y TEST
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

                # CREAMOS LAS PIPELINES PARA CADA MODELO
                pipeline1 = Pipeline([
                    ('scaler', StandardScaler()),  # Optional: Scale features
                    ('regressor', DecisionTreeRegressor())  # Decision tree regressor
                ])

                pipeline2 = Pipeline([
                    ('scaler', StandardScaler()),  # Optional: Scale features
                    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))  
                ])
                pipeline3 = Pipeline([
                    ('scaler', StandardScaler()),  #cRe escalado de columnas
                    ('regressor', KNeighborsRegressor(n_neighbors=5))  # Decision KNN regressor
                ])

                pipeline4 = Pipeline([
                    ('scaler', StandardScaler()),  #Re escalado de columnas
                    ('regressor', LinearRegression())  # Linear regressor
                ])
                # TRAIN
                pipeline1.fit(X_train, y_train)
                pipeline2.fit(X_train, y_train)
                pipeline3.fit(X_train, y_train)
                pipeline4.fit(X_train, y_train)

                # PREDICT
                y_pred1 = pipeline1.predict(X_test)
                y_pred2 = pipeline2.predict(X_test)
                try:
                    y_pred3 = pipeline3.predict(X_test)
                except ValueError: #Esto lo agregué porque a veces si hay muy pocos resultados tiraba este error
                    y_pred3 = 0
                y_pred4 = pipeline4.predict(X_test)

                # CALCULO DE METRICAS
                print("\n")
                print("Metricas de los 4 modelos utilizados")
                print("\n")

                rmse1 = mean_squared_error(y_test, y_pred1, squared=False)
                print("Tree Regressor RMSE: ", rmse1)
                r2_score1 = r2_score(y_test, y_pred1)
                print("Tree Regressor R-Squared: ",r2_score1)
                print("\n")

                rmse2 = mean_squared_error(y_test, y_pred2, squared=False)
                print("Random Forest RMSE: ", rmse2)
                r2_score2 = r2_score(y_test, y_pred2)
                print("Random Forest R-Squared: ",r2_score2)
                print("\n")

                rmse3 = mean_squared_error(y_test, y_pred3, squared=False)
                print("KNN RMSE: ",rmse3)
                r2_score3 = r2_score(y_test, y_pred3)
                print("KNN R-Squared: ",r2_score3)
                print("\n")

                rmse4 = mean_squared_error(y_test, y_pred4, squared=False)
                print("Linear Regressor RMSE: ", rmse4)
                r2_score4 = r2_score(y_test, y_pred4)
                print("Linear Regressor R-Squared: ",r2_score4)
                print("\n")

                performance = pd.DataFrame(data={'RMSE':[rmse1,rmse2,rmse3,rmse4],'R-Squared':[r2_score1,r2_score2,r2_score3,r2_score4]},index=["Tree Reg","RF Reg","Knn Reg","Linear Reg"])

                #Si no se ingresaron datos para la prediccion se utiliza el promedio de kms y los valores mas comunes de DBautosinoutliers
                if DatosPredecir.loc[0,'awd'] == None:
                    DatosPredecir.loc[0,'awd'] = DBautosinoutliers['awd'].value_counts().index[0]
                if DatosPredecir.loc[0,'kms'] == None:
                    DatosPredecir.loc[0,'kms'] = DBautosinoutliers['kms'].mean()
                if DatosPredecir.loc[0,'manual'] == None:
                    DatosPredecir.loc[0,'manual'] = DBautosinoutliers['manual'].value_counts().index[0]
                if DatosPredecir.loc[0,'motor'] == None:
                    DatosPredecir.loc[0,'motor'] = DBautosinoutliers['motor'].value_counts().index[0]
                if DatosPredecir.loc[0,'nafta'] == None:
                    DatosPredecir.loc[0,'nafta'] = DBautosinoutliers['nafta'].value_counts().index[0]

                
                #LAS PREDICCIONES
                ypred1 = pipeline1.predict(DatosPredecir)
                ypred2 = pipeline2.predict(DatosPredecir)
                ypred3 = pipeline3.predict(DatosPredecir)
                ypred4 = pipeline4.predict(DatosPredecir)

                # Calcular los cuatro valores obtenidos por los modelos predictivos
                valor_1 = int(ypred1[0])
                valor_2 = int(ypred2[0])
                valor_3 = int(ypred3[0])
                valor_4 = int(ypred4[0])
                mediana = (valor_1 + valor_2 + valor_3 + valor_4) / 4
                

                # Crear el gráfico precio vs cantidad
                
                ax = sns.histplot(data=DBauto,x='precio')
                ax.axvline(precio_mean, color='black', linestyle='--', label='Promedio')
                ax.axvline(mediana, color='red', linestyle='-', label='Prediccion')

        else: #SI HAY MENOS DE 10 RESULTADOS ENTONCES NO SE GENERAN PREDICCIONES
                ypred1=[0]
                ypred2=[0]
                ypred3=[0]
                ypred4=[0]
                rmse1=0
                r2_score1=0
                rmse2=0
                r2_score2=0
                rmse3=0
                r2_score3=0
                rmse4=0
                r2_score4=0
                performance=0
                #se grafica solo las estadisticas
                sns.set()
                ax = sns.histplot(data=DBauto,x='precio')
                ax.axvline(precio_mean, color='red', linestyle='-', label='Promedio')
                mediana = 0
        
        #Formateo del grafico haya o no resultados
        
        ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        ax.legend(loc='upper left',bbox_to_anchor=(1, 1.1))
        ax.set(xlim=(lower*0.95, upper*1.05))
        plt.ylabel('Cantidad')
        plt.xlabel('Precio(en USD)')
        plt.title(marca.capitalize()+" - "+modelo.capitalize()+" ( "+str(anio)+" )")

        # Guarda el gráfico precio vs cantidad
        sns.despine()
        
        temp_file = 'static/temp_file.png'
        plt.savefig(temp_file, bbox_inches='tight')
        plt.close()
    
        hist1 = grafcantidadhist(marca.capitalize(),modelo.capitalize(),int(anio))
        hist2 = grafpreciohist(marca.capitalize(),modelo.capitalize(),int(anio))
        # Render the prediction result template with the predicted price
        return render_template('result.html',
                    cantidad=cantidad,
                    tamanioDB=tamanioDB,
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
                    ypred3=ypred3[0],
                    ypred4=ypred4[0],
                    DBautosbaratos=DBautosbaratos,
                    DBbuenprecioxkm=DBbuenprecioxkm,
                    rmse1=rmse1,
                    r2_score1=r2_score1,
                    rmse2=rmse2,
                    r2_score2=r2_score2,
                    rmse3=rmse3,
                    r2_score3=r2_score3,
                    rmse4=rmse4,
                    r2_score4=r2_score4,
                    plot_path=temp_file,
                    versionFREQ=versionFREQ,
                    performance=performance,
                    hist1 = hist1,
                    hist2 = hist2,
                    mediana = mediana 
                    )
    else:
         return render_template('zero.html')    #UN TEMPLATE SI NO HUBO RESULTADOS

# Borra los csv viejos si tienen mas de 
cleanup_csv_files('./data/raw/')

if __name__ == '__main__':
    #app.run(debug=True, )
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



        












