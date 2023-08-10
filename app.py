from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from functions import round_next, load_car_models, get_marcas, grafpreciohist,grafcantidadhist, DBauto_funcion, get_dolarhoy,modelado, grafvalorcantidad, modelado
import os
import re
from thefuzz import fuzz
from sklearn.model_selection import train_test_split

# OBTENIENDO marcas y modelos (desde una funcion que toma un csv y un txt donde estan estos valores)
car_models = load_car_models() #diccionario de modelos
car_brands = get_marcas()  #lista de marcas


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

#API REQUEST y Obtencion de datos a dataframe
    DBauto = DBauto_funcion(marca,modelo,anio,'APP_USR-1735483845190568-052521-4e03162aa48b46d277ba15d55d6a2357-224513779')
    
    #Obtener valor del dolar del dia
    dolarhoy = get_dolarhoy()
    print(f'primer len {len(DBauto)}')

    #LImpieza y transformación de features (columnas)
    #Estos primeros pasos son para eliminar modelos y años que no sean los solicitados, ya que mercadolibre a veces incluye otros resultados.
    
    #elimina otros años
    DBauto = DBauto[DBauto['año'] == str(anio)]
    print(f'seg len {len(DBauto)}')

    #hace las columnas categoricas solo de 0 y 1
    DBauto['nafta'] = DBauto['nafta'].replace(['Nafta','nafta'],1).replace(to_replace=r'^((?!1).)*$', value=0, regex=True).fillna(0).astype(int)
    DBauto['awd'] = DBauto['awd'].replace(['Integral','4x4'],1).replace(to_replace=r'^((?!1).)*$', value=0, regex=True).fillna(0).astype(int)
    DBauto['manual'] = DBauto['manual'].replace(['Manual','manual'],1).replace(to_replace=r'^((?!1).)*$', value=0, regex=True).fillna(0).astype(int)
    print(f'ter len {len(DBauto)}')

    #crea col para comparar modelo solicitado vs modelos en la busqueda
    DBauto['m%'] = DBauto['modelo'].apply(lambda x: fuzz.ratio(str(x),str(modelo)))
    print(f'cuar len {len(DBauto)}')
    #elimina otros modelos
    DBauto = DBauto[DBauto['m%'] > 80]
    print(f'quin len {len(DBauto)}')
    #actualiza cantidad de resultados reales del modelo y año seleccionado
    cantidad = len(DBauto)
    
    
    #SI LA BUSQUEDA ARROJA RESULTADOS ENTONCES SE EJECUTA el CODIGO, sino te avisa que no hay y volves a empezar
    if len(DBauto) > 0:
        DBauto.sort_values(by=['version'])
        DBauto.reset_index(drop=True, inplace=True) #reset index porque generaba problemas al buscar por filas inexistentes
        DBauto.fillna(method='ffill',inplace=True)
        #KMS 
        #texto a int
        if DBauto['kms'].dtype != int:
            DBauto['kms'] = DBauto['kms'].str.extract('(\d+)').astype(int)


        #PRECIO 
        #dolares a pesos utilizando la cotizacion dolarhoy blue
        for i,row in DBauto.iterrows():
                x = int(row['precio'])
                if row['moneda'] == 'ARS':
                        DBauto.at[i,'precio'] = x/dolarhoy
                        DBauto.at[i,'moneda'] = 'USD'


        #MOTOR
        #se encontraron muchos modelos que estaban cargados como centimetros cubicos con valores extraños como 1206cc por lo que tuve que armar una funcion que normalice los datos
        if DBauto['motor'].dtype == int:
                DBauto['motor'].astype(float)
        elif DBauto['motor'].dtype == str or DBauto['motor'].dtype == object:
                DBauto['motor'] = DBauto['motor'].str.extract('(\d+(?:\.\d+)?)').astype(float)
        
        for i,row in DBauto.iterrows():
                if row['motor'] > 12.0:         #busca motores en cc y conviertes a litros
                        DBauto.at[i,'motor'] = round_next(DBauto.at[i,'motor'] / 1000)
        
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

        #DB SOLO de los autos entre precios normales de lower y upper
        #Es la DB final y mas importante ya que elimina valores fuera del rango normal
        DBautosinoutliers = DBauto[(DBauto['precio'] >= lower) & (DBauto['precio'] <= upper)].copy(deep=True)

        #Genero DBs de promedio para cada version, cada motor, cada traccion y cada combustible
        meanFuel = DBautosinoutliers.groupby('nafta',as_index=False)['precio'].mean()
        meanFuel['precio'] = meanFuel['precio'].apply(lambda x: f'${x:.0f}')
        meanFuel['nafta'] = meanFuel['nafta'].replace([0,'0'],'Diesel').replace([1,'1'],'Nafta')
        meanFuel.rename(columns={'nafta':'Combustible','precio':'Precio'},inplace=True)
        meanFuel.set_index('Combustible',inplace=True)
        
        meanVersion = DBautosinoutliers.groupby('version',as_index=False)['precio'].mean()
        meanVersion['precio'] = meanVersion['precio'].apply(lambda x: f'${x:.0f}')
        meanVersion.rename(columns={'version':'Version','precio':'Precio'},inplace=True)
        meanVersion.set_index('Version',inplace=True)

        meanMotor = DBautosinoutliers.groupby('motor',as_index=False)['precio'].mean()
        meanMotor['precio'] = meanMotor['precio'].apply(lambda x: f'${x:.0f}')
        meanMotor.rename(columns={'motor':'Motor','precio':'Precio'},inplace=True)
        meanMotor.set_index('Motor',inplace=True)

        meanAwd = DBautosinoutliers.groupby('awd',as_index=False)['precio'].mean()
        meanAwd['precio'] = meanAwd['precio'].apply(lambda x: f'${x:.0f}')
        meanAwd['awd'] = meanAwd['awd'].replace([0,'0'],'4x2').replace([1,'1'],'4x4')
        meanAwd.rename(columns={'awd':'Traccion','precio':'Precio'},inplace=True)
        meanAwd.set_index('Traccion',inplace=True)

        meanManual = DBautosinoutliers.groupby('manual',as_index=False)['precio'].mean()
        meanManual['precio'] = meanManual['precio'].apply(lambda x: f'${x:.0f}')
        meanManual['manual'] = meanManual['manual'].replace([0,'0'],'Autom.').replace([1,'1'],'Manual')
        meanManual.rename(columns={'manual':'Transmision','precio':'Precio'},inplace=True)
        meanManual.set_index('Transmision',inplace=True)


        #Autos baratos por precio
        DBautosbaratos = DBauto[(DBauto['precio'] < lower)].copy(deep=True)
        DBautosbaratos = DBautosbaratos[['version','motor','precio','kms','precioxkm','link']].drop_duplicates(subset='link',keep='first')

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
        DBbuenprecioxkm = DBbuenprecioxkm[['version','motor','precio','kms','precioxkm','link']].drop_duplicates(subset='link',keep='first')

        #stats 
        precio_mean = DBautosinoutliers['precio'].mean()
        precio_med = DBautosinoutliers['precio'].median()
        kms_mean = DBautosinoutliers['kms'].mean()
        kms_med = DBautosinoutliers['kms'].median()


        #Versiones mas frecuentes segun la busqueda
        versionFREQ = DBautosinoutliers['version'].value_counts().rename_axis('Version').reset_index(name='Cantidad')
    
        # Crea dataframe con las columnas que van a usarse para la prediccion
        DatosPredecir = pd.DataFrame(columns=['motor','nafta','kms','awd','manual'])

        #Se utiliza el promedio de kms y los valores mas comunes de DBautosinoutliers
        DatosPredecir.loc[0,'awd'] = DBautosinoutliers['awd'].value_counts().index[0]
        DatosPredecir.loc[0,'kms'] = DBautosinoutliers['kms'].mean()
        DatosPredecir.loc[0,'manual'] = DBautosinoutliers['manual'].value_counts().index[0]
        DatosPredecir.loc[0,'motor'] = DBautosinoutliers['motor'].value_counts().index[0]
        DatosPredecir.loc[0,'nafta'] = DBautosinoutliers['nafta'].value_counts().index[0]

        DPtrac = '4x2' if DatosPredecir.loc[0,'awd'] == 0 else '4x4'
        DPkms = DatosPredecir.loc[0,'kms'].astype(int)
        DPtrans = 'Autom' if DatosPredecir.loc[0,'manual'] == 0 else 'Manual'
        DPmotor = DatosPredecir.loc[0,'motor']
        DPcombus = 'Diesel' if DatosPredecir.loc[0,'nafta'] == 0 else 'Nafta'
        
        tamanioDB = len(DBautosinoutliers)
        
        if tamanioDB > 10: #SI HAY MAS DE 10 RESULTADOS EN LA BUSQUEDA SE GENERAN LAS PREDICCIONES
            #MODELADO
            #armamos la DB y separamos en X e y
            Processedx = DBautosinoutliers[['motor','nafta','precio','kms','awd','manual']].copy(deep=True)
            X = Processedx.drop('precio',axis=1)
            y = Processedx['precio']
            # DIVIDIMOS LOS DATOS EN DOS SETS TRAIN Y TEST
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

            valor_1,rmse1,r2_score1 = modelado(X_train, X_test, y_train, y_test,'tree',DatosPredecir)
            valor_2,rmse2,r2_score2 = modelado(X_train, X_test, y_train, y_test,'rf',DatosPredecir)
            valor_3,rmse3,r2_score3 = modelado(X_train, X_test, y_train, y_test,'knn',DatosPredecir)
            valor_4,rmse4,r2_score4 = modelado(X_train, X_test, y_train, y_test,'linear',DatosPredecir)
            prediccion = (valor_1 + valor_2 + valor_3 + valor_4) / 4
            performance = pd.DataFrame(data={'RMSE':[rmse1,rmse2,rmse3,rmse4],'R-Squared':[r2_score1,r2_score2,r2_score3,r2_score4]},index=["Tree Reg","RF Reg","Knn Reg","Linear Reg"])

                
                
                
        else: #SI HAY MENOS DE 10 RESULTADOS ENTONCES NO SE GENERAN PREDICCIONES
            valor_1=0
            valor_2=0
            valor_3=0
            valor_4=0
            rmse1=0
            r2_score1=0
            rmse2=0
            r2_score2=0
            rmse3=0
            r2_score3=0
            rmse4=0
            r2_score4=0
            performance=0
            prediccion = 0
        
       
        #wik=get_wikipedia_link(marca,modelo)
        #intro = wiki_intro(wik)
        #primera = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s", intro)[0]
        #foto = wiki_foto(wik,f'./static/{marca}{modelo}.jpg')
       #GRAFICOS 
        grafico = grafvalorcantidad(DBautosinoutliers,precio_mean,prediccion,marca,modelo,anio)
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
                    ypred1=valor_1,
                    ypred2=valor_2,
                    ypred3=valor_3,
                    ypred4=valor_4,
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
                    plot_path=grafico,
                    versionFREQ=versionFREQ,
                    performance=performance,
                    hist1 = hist1,
                    hist2 = hist2,
                    mediana = prediccion,
                    meanFuel = meanFuel,
                    meanVersion = meanVersion,
                    meanMotor = meanMotor,
                    meanAwd = meanAwd,
                    meanManual = meanManual,
                    DPtrac = DPtrac,
                    DPcombus = DPcombus,
                    DPkms = DPkms,
                    DPtrans = DPtrans,
                    DPmotor = DPmotor
                    )
    else:
         return render_template('zero.html')    #UN TEMPLATE SI NO HUBO RESULTADOS


if __name__ == '__main__':
    #app.run(debug=True, )
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))



        












