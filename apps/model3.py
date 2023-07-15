import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yfinance as yf

def app():
    st.title('Model - SVC')

    #start = '2004-08-18'
    #end = '2022-01-20'

    start = '2018-1-1'
    end = '2023-1-1'
    #start = st.date_input('Start' , value=pd.to_datetime('2004-08-18'))
    #end = st.date_input('End' , value=pd.to_datetime('today'))

    st.title('Predicción de tendencia de acciones')

    user_input = st.text_input('Introducir cotización bursátil' , 'GOOG')

    datap = yf.download(user_input, start , end)
    # df = data.DataReader(user_input, start, end)

    # Describiendo los datos

    st.subheader('Datos del 2018 al 2023') 
    st.write(datap.describe())
    
    # Candlestick chart
    st.subheader('Gráfico Financiero') 
    candlestick = go.Candlestick(
                            x=datap.index,
                            open=datap['Open'],
                            high=datap['High'],
                            low=datap['Low'],
                            close=datap['Close']
                            )

    fig = go.Figure(data=[candlestick])

    fig.update_layout(
        width=800, height=600,
        title=user_input,
        yaxis_title='Precio'
    )
    
    st.plotly_chart(fig)
    
    
    

    # Añadiendo indicadores para el modelo
    datap['Open-Close'] = datap.Open - datap.Close
    datap['High-Low'] = datap.High - datap.Low
    
    
    # Modelo SVC
    
    ## Variables predictoras
    X = datap[['Open-Close', 'High-Low']]
    ## Variable objetivo
    y = np.where(datap['Close'].shift(-1) > datap['Close'], 1, 0)
    ## División data de entrenamiento y prueba
    split_percentage = 0.8
    split = int(split_percentage*len(datap))
    ## Entrenando el dataset
    X_train = X[:split]
    y_train = y[:split]
    ## Testeando el dataset
    X_test = X[split:]
    y_test = y[split:]
    ## Creación del modelo
    cls = svm.SVC(probability=True).fit(X_train, y_train)
    ## Predicción del test
    y_pred = cls.predict(X_test)
    
    
    # Señal de predicción 
    
    datap['Predicted_Signal'] = cls.predict(X)
    ## Añadiendo columna condicional
    conditionlist = [
    (datap['Predicted_Signal'] == 1) ,
    (datap['Predicted_Signal'] == 0)]
    choicelist = ['Comprar','Vender']
    datap['Decision'] = np.select(conditionlist, choicelist)
    st.subheader('Predicción de señal de compra o venta') 
    st.write(datap)    
    
    
    
    # Estrategia de Implementación
    
    # Cálculo de las devoluciones diarias
    datap['Return'] = datap.Close.pct_change()
    # Cálculo de los rendimientos de la estrategia
    datap['Strategy_Return'] = datap.Return*datap.Predicted_Signal.shift(1)
    # Cálculo de los rendimientos acumulativos
    datap['Cum_Ret'] = datap['Return'].cumsum()
    # Cálculo de los rendimientos acumulativos de la estrategia
    datap['Cum_Strategy'] = datap['Strategy_Return'].cumsum()
    # Retornos de la estrategia de trama vs rendimientos originales
    st.subheader('Retornos de la estrategia de trama vs. Rendimientos originales') 
    fig = px.line(datap,y=['Cum_Ret', 'Cum_Strategy'])
    st.plotly_chart(fig)


    # Visualizando la data 3
    st.subheader('Precio predecido vs Precio Original')
    fig9=plt.figure(figsize=(12,6))
    plt.plot(datap['Cum_Ret'],color='blue' , label = 'Precio Original')
    plt.plot(datap['Cum_Strategy'],color='red' ,label= 'Precio Predecido')
   # plt.plot(y_test, 'b', label = 'Precio Original')
   # plt.plot(predictions, 'r', label= 'Precio Predecido')
    plt.xlabel('Tiempo')
    plt.ylabel('Precio')
    plt.legend()
    st.pyplot(fig9)


   # plt.plot(datap['Cum_Ret'],color='red')
   # plt.plot(datap['Cum_Strategy'],color='blue')
    
    
    










    # Evaluación del modelo
    
    st.title('Evaluación del Modelo SVC')
    ## Matriz de confusión
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
    st.subheader('Matriz de confusión') 
    st.write(cm)
    ## Métricas
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred)
    sensivity = metrics.recall_score(y_test, y_pred)
    f1_score = metrics.f1_score(y_test, y_pred)
    metricas = {
    'metrica' : ['Exactitud', 'Precisión', 'Sensibilidad','F1-score'],
    'valor': [accuracy, precision, sensivity,f1_score]
    }
    metricas = pd.DataFrame(metricas)  
    ### Gráfica de las métricas
    st.subheader('Métricas de rendimiento') 
    fig = px.bar(        
        metricas,
        x = "metrica",
        y = "valor",
        title = "Métricas del modelo SVC",
        color="metrica"
    )
    st.plotly_chart(fig)
    ## Curva ROC
    predictions = cls.predict_proba(X_test)
    predictions = predictions[:, 1]
    cls_fpr, cls_tpr, threshold = roc_curve(y_test, predictions)
    auc_cls = auc(cls_fpr, cls_tpr)
    roc = pd.DataFrame({'fpr': cls_fpr, 'tpr': cls_tpr})
    ### Gráfica ROC
    st.subheader('ROC Curve') 
    ### AUC
    st.write('Support Vector Classifier: ROC AUC=%.3f' % (auc_cls))
    fig = px.line(
    roc,    
    x = "fpr",
    y = "tpr"
    )
    st.plotly_chart(fig)