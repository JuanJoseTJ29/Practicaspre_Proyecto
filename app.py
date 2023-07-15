import streamlit as st
from multiapp import MultiApp
from apps import home, model, model2  

#modelRF  
# import your app modules here model2

app = MultiApp()

st.markdown("""
#  PROYECTO FINAL PRACTICAS PRE PROFESIONALES


""")

# Add all your application here
app.add_app("Home", home.app)
app.add_app("Modelo LSTM", model.app)
#app.add_app("Modelo SVR", model2.app)
# app.add_app("Modelo Random Forest", modelRF.app)
app.add_app("Modelo SVC", model2.app)
# The main app
app.run()



