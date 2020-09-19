# -*- coding: utf-8 -*-
"""
Created on Sat Sep 12 11:46:53 2020

@author: manuelquintana
"""

import pandas as pd
from sklearn import preprocessing
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


route = 'D:/FACULTAD/Data science/TP3/'
df_indicators = pd.read_excel(route + 'input - variables.xlsx', index_col=0)

'''
 Data cleaning
'''
columns = ['X_auto_Mex', 'X_auto_Arg', 'X_Auto_Bra', 'X_total_BRA']
df_indicators = df_indicators.drop(columns, axis = 1)
# La variable "Global PMI" tiene nulos para todo el año 2007
# Los rellenamos con la medIA: 50.4905
df_indicators['GLOBAL PMI'] = df_indicators['GLOBAL PMI'].fillna(df_indicators['GLOBAL PMI'].mean())
# La variable RWI/ISL-Container-Throughput-Index tiene nan en el ultimo mes
previous_value = df_indicators.loc['2020-07-01']['RWI/ISL-Container-Throughput-Index']
df_indicators.iloc[-1,5] = previous_value


'''
Data Normalization
# Esto nos permite por un lado graficar en la misma escala todas las 
# variables, y por otro, aplicar transformaciones sin que de error 
# por ejemplo calcular el log de números negativos.
'''
# Normalizamos para tener la misma escala y graficar tendencias
x = df_indicators.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df_normalized = pd.DataFrame(x_scaled)
# Reconstruimos los ejes: fechas y columnas
df_normalized = df_normalized.set_axis(df_indicators.index)
df_normalized = df_normalized.set_axis(df_indicators.columns, axis = 'columns')

'''
# Graficos de tendencias anuales (todas las variables)
for j in df_normalized.columns:    
    for anio in range(2007, 2020):
        ax = df_normalized.loc[str(anio), j].plot()
        ax.set_ylabel('Columnas');
        ax.set_xlabel('Anios');
'''


# Probar transformaciones: 
# Usaremos 2: 
# BoxCox: 
# Logit:  logit(p) = log(p/(1-p))


# Primera transformacion
from sklearn.preprocessing import power_transform
df_normalized = df_normalized.replace(0, 0.00001)
transf_boxcox = power_transform(df_normalized, method='box-cox')
df_boxcox = pd.DataFrame(data = transf_boxcox)
df_boxcox.plot()
df_normalized = df_normalized.replace(0.00001, 0)

# Esta es la misma transformacion pero con otro metodo
#transf_yeo = power_transform(df_indicators, method='yeo-johnson')
#dfyeo = pd.DataFrame(data = transf_yeo)
#dfyeo.plot()

# Esta es la misma transformacion pero estandarizada, solo se mueven los ejes pero la curva es igual
#from sklearn.preprocessing import PowerTransformer
#power = PowerTransformer(method='yeo-johnson', standardize=True)
#data_trans = power.fit_transform(dfyeo)
#dfyeostandardized = pd.DataFrame(data = data_trans)
#dfyeostandardized.plot()


# Segunda transformacion
from scipy.special import logit
transf_logit = logit(df_normalized['PC_AGRO'])
dflogit = pd.DataFrame(data = transf_logit)
#plt.grid(color='r', linestyle='-', linewidth=0.5)
dflogit.plot()


import math
# Histograma Original
plt.hist(df_normalized['PC_AGRO'], bins = len(df_normalized['PC_AGRO']))
plt.show()
# Logit con valor absoluto!!!! --> Falta normalizar entre 0 y 1 el valor de P
df_normalized = df_normalized.replace(0, 0.00001)
plt.grid(color='r', linestyle='-', linewidth=0.5)
plt.hist(np.log(abs(df_normalized['PC_AGRO'] / (1-df_normalized['PC_AGRO']))), bins = len(df_normalized['PC_AGRO']))
plt.show()




for i in [1,2,3,4,5]:    
    pyplot.hist((df_indicators['PC_AGRO'])**(1/i), bins=40)
    pyplot.title("Transformation 1/{}".format(str(i))) # i is a integer, so we must convert it to a string value
    

    
'''
Busqueda de outliers:
Utilizamos dos tecnicas:
1. Definimos un rango intercuartil y detectamos los valores que caen fuera.
2. Con las graficas Boxplot detectamos visualmente qué columnas presentan outliers.
3. Otra opcion seria analizar la curtosis para saber cuan plana o crespada es la distribución de cada variable.
'''
# Metodo 1
def detect_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.01)
    q2 = df_in[col_name].quantile(0.99)
    #iqr = q2-q1 #Interquantile range
    #fence_low  = q1 - 1.5 * iqr
    #fence_high = q2 + 1.5 * iqr
    fence_low  = q1
    fence_high = q2
    df_out = df_in
    for index, row in df_out.iterrows():
        if (row[col_name] <= fence_low) or (row[col_name] >= fence_high):
            print(f'Outlier detectado: Columna:{col_name}/ Valor: {row[col_name]}\nFence_low: {fence_low}/Fence_high: {fence_high}')
            #row[col_name] = df_out[col_name].mean()
    return df_out
for col_name in df_indicators.columns:
    df_indicators = detect_outlier(df_indicators, col_name)

# Metodo 2
for col_name in df_indicators.columns:
    sns.boxplot(x=df_indicators[col_name], orient='h')
    plt.figure()

 
'''
Limpieza de outliers
'''
# Detectamos una anomalia en el mes 4 cuando comenzo la cuarentena en 
# Argentina, Brasil y Mexico, las fabricas de autos cerraron y la produccion bajo a 0
# Reemplazaremos el valor del mes 4 y mes 5 de 2020 por el valor del mes siguiente
# al reactivar la actividad plenamente.
# !!!!!!!!!!!!!!!!!!!!!!!!!
# Probar con la mediana hasta el periodo anterior a los outliers
# !!!!!!!!!!!!!!!!!!!!!!!!!
#Argentina
next_value = df_indicators.loc['2020-06-01']['Produccion vehiculos Argentina']
df_indicators['Produccion vehiculos Argentina']['2020-05-01'] = next_value
df_indicators['Produccion vehiculos Argentina']['2020-04-01'] = next_value
#Brasil
next_value = df_indicators.loc['2020-06-01']['Produccion vehiculos Brazil']
df_indicators['Produccion vehiculos Brazil']['2020-05-01'] = next_value
df_indicators['Produccion vehiculos Brazil']['2020-04-01'] = next_value
#Mexico
next_value = df_indicators.loc['2020-06-01']['Produccion Autos Mexico']
df_indicators['Produccion Autos Mexico']['2020-05-01'] = next_value
df_indicators['Produccion Autos Mexico']['2020-04-01'] = next_value

'''
Analizar outliers en:
PC_METALS
PC_CURRENCY
PC_INDEX
PRODUCCION AUTOS MEXICO
PRODUCCION VEHICULOS Brazil
PRODUCCION VEHICULOS Argentina
BDI
IFO = Business Climate
IFO = Business Expectations
USMPI
CHINA NON MANUFACTURING PMI
CHINA MANUFACTURING PMI
GLOBAL PMI
PMI MEX
'''
  
# Construimos una matriz de correlacion
# df_correlated = round(df_indicators.corr(),3)
df_correlated_normalized = round(df_normalized.corr(),3)

# Nos quedamos con las variables que se correlacionan en mas de un 60%
# Sin normalizar
''' Observacion: ya sea que previamente hayamos normalizado o no, los resultados
    en la matriz de correlation son los mismos!!
'''
#df_upper = df_correlated.where(np.triu(np.ones(df_correlated.shape),k=1).astype(np.bool))
# Normalizada
df_upper = df_correlated_normalized.where(np.triu(np.ones(df_correlated_normalized.shape),k=1).astype(np.bool))
for i in df_upper.columns:
    for j in range(len(df_upper)):
        if (df_upper[i][j] > -0.6 and df_upper[i][j] < 0.6):
           df_upper[i][j] = np.nan

# Variables que queremos predecir y variables con las que mejor se relacionan:
# 1. PC_AGRO: pc_energy, produccion vehiculos argentina, Bloomberg, Thomsonreuters
# 2. PC_ENERGY: PRODUCCION vehiculos brazil, produccion vehiculos argentina, Bloomberg, Thomsonreuters
# 3. PC_METALS: BLOOMBERG, Thomsonreuters, BDI BALTIC DRY INDEX,. CHINA NON MANUFACTURING
# 4. PC_CURRENCY: PC_INDEX, RWI, produccion vehiculos argentina, Bloomberg, Thomsonreuters
   
df_indicators = df_indicators[['PC_AGRO', 'PC_ENERGY', 'PC_METALS', 
                              'PC_CURRENCY', 'PC_INDEX',
                              'Produccion vehiculos Argentina',
                              'Produccion vehiculos Brazil',
                              'RWI/ISL-Container-Throughput-Index',
                              'Bloomberg Commodity Index',
                              'Thomson Reuters/CoreCommodity CRB Commodity Index',
                              'BDI = Baltic Dry Index',
                              'CHINA NON MANJUFACTURING PMI'                    
                              ]]

df_indicators = df_indicators.rename(columns = {
            'Produccion vehiculos Argentina': 'VEH_ARG',
            'Produccion vehiculos Brazil': 'VEH_BRA',
            'RWI/ISL-Container-Throughput-Index': 'RWI',
            'Bloomberg Commodity Index': 'BCI',
            'Thomson Reuters/CoreCommodity CRB Commodity Index': 'CRB',
            'BDI = Baltic Dry Index': 'BDI',
            'CHINA NON MANJUFACTURING PMI': 'CNM_PMI'})

# Posibles outliers:
outliers = ['PC_METALS',
            'PC_CURRENCY',
            'PC_INDEX',
            'VEH_BRA',
            'VEH_ARG',
            'BDI',
            'CNM_PMI']

# Graficamos la linea de esas variables:
for po in outliers:
    df_indicators[po].plot(title = po)
    plt.figure()
    
'''
En todos los casos vemos dos caídas grandes:
1. A mediados de 2008 y hasta mediados de 2009 producto de la crisis económica en USA
2. Entre marzo de 2020 y abril de 2020 producto del Covid Outbreak
La primera tuvo origen económico, por lo que se relaciona con el mundo
que estamos analizando.
En cambio la segunda es de origen sanitario y queda excluida.
'''
    
'''
Detectamos outliers en:
    -Produccion vehículos Brazil (2009 y 2020)
    -Produccion vehiculos Argentina (2009 y 2020)
    -BDI = Baltic Dry Index (2007 Y 2008)
    -CHINA NON MANJUFACTURING PMI (2020)
'''




