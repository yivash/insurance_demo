import streamlit as st
import pandas as pd
from joblib import dump, load
import pickle
from sklearn.ensemble import RandomForestRegressor
import os


#from sklearn import datasets
#import xgboost as xg


st.write("""
# Demo Insurance Regression Prediction App
""")

st.sidebar.header('User Input Parameters')



def user_input_features():
    isweekend = st.sidebar.slider('isweekend', 0, 1, 0)
    month = st.sidebar.slider('month', 1, 12, 6)
    season = st.sidebar.slider('season', 1, 4, 2)
    Notification_period = st.sidebar.slider('Notification_period', 0, 150, 20)
    Inception_to_loss = st.sidebar.slider('Inception_to_loss', 1, 300, 100)
    Weather_conditions = st.sidebar.slider('Weather_conditions', 0, 2, 0)
    Time_hour = st.sidebar.slider('Time_hour', 0, 23, 8)
    TP_injury_traumatic = st.sidebar.slider('TP_injury_traumatic', 0, 1, 0)
    TP_injury_fatality = st.sidebar.slider('TP_injury_fatality', 0, 1, 0)
    sum_TP = st.sidebar.slider('sum_TP', 0, 5, 0)
    
    data = {'isweekend': isweekend,
            'month': month,
            'season': season,
            'Notification_period': Notification_period,
            'Inception_to_loss': Inception_to_loss,
            'Weather_conditions': Weather_conditions,
            'Time_hour': Time_hour,
            'TP_injury_traumatic': TP_injury_traumatic,
            'TP_injury_fatality': TP_injury_fatality,
            'sum_TP': sum_TP
            }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

cwd = os.getcwd()

if not cwd.endswith("src"):
    src_dir = os.path.join(cwd, "src")
else:
    src_dir = cwd

st.write(src_dir)

# List all files in the directory
files = os.listdir(src_dir)
st.write(files)

model_path = os.path.abspath("rf_model.pkl")
st.write(model_path)

model = load(model_path)

prediction = model.predict(df)

st.subheader('Prediction')
st.write(prediction)
