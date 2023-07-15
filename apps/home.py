import streamlit as st
#ruta_imagen = "ruta/a/imagen.jpg"

def app():
    st.title('Alumno:')
    st.write('Tirado Julca , Juan Jose - 18200117')

    st.title('Descripcion del proyecto:')
    st.write('El proyecto consiste en implementar un algoritmo de predicción de precios de acciones basado en los modelos LSTM y SVC que permita a inversionistas individuales u administradores mejorar la rentabilidad de sus inversiones, puesto que la predicción precisa de los precios de los instrumentos financieros es esencial para tomar mejores decisiones de inversiones con mínimo riesgo. Y con la realización del sistema web podrán visualizar el análisis y resultados obtenidos.')
  #  st.image(ruta_imagen, caption='Descripción de la imagen', use_column_width=True)

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://static.vecteezy.com/system/resources/previews/008/884/464/non_2x/trend-up-graph-icon-stock-sign-growth-progress-red-arrow-icon-line-chart-symbol-vector.jpg");
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
        
        )