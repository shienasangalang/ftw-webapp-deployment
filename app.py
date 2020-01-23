import pandas as pd
import streamlit as st
import joblib
import numpy as np

# Title of your Web Application
st.title('Sales Forecasting')

# Describe your webapp
st.write('We demonstrate how we can forecast advertising sales based on ad expenditure.')

# Read Data
data = pd.read_csv('Data/advertising_regression.csv')

# Show Data
data

# Create Sidebar
# Sidebar Description
st.sidebar.subheader('Advertising Costs')

# TV slider
TV = st.sidebar.slider('TV Advertising Cost', 0, 300, 150)

# Radio slider
Radio = st.sidebar.slider('Radio Advertising Cost', 0, 50, 25)

# Newspaper slider
Newspaper = st.sidebar.slider('Newspaper Advertising Cost', 0, 250, 125)

# TV
st.subheader('TV Advertising Cost Distribution')

# Distribution of Radio Advertising Cost
hist_values = np.histogram(data.TV, bins = 300, range = (0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

# RADIO
st.subheader('Radio Advertising Cost Distribution')

# Distribution of Radio Advertising Cost
hist_values = np.histogram(data.radio, bins = 300, range = (0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

# NEWSPAPER
st.subheader('Newspaper Advertising Cost Distribution')

# Distribution of Radio Advertising Cost
hist_values = np.histogram(data.newspaper, bins = 300, range = (0,300))[0]

# Show Bar Chart
st.bar_chart(hist_values)

# load Saved Machine Learning Model
saved_model = joblib.load('advertising_model.sav')

# Predict Sales using Variables/Features
predicted_sales = saved_model.predict([[TV, Radio, Newspaper]])[0]

# Print Predictions
st.write(f"Predicted Sales is {predicted_sales} dollars.")
# f is for format