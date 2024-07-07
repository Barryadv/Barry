# in command line in my environment,
# C:\Users\barry\Documents\GitHub\Be-P streamlit run  br_app.py  
#conda activate Barryadv not necessary because have only one env't base(root)
#use markdown cheatsheet by adam pritchard

import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import statsmodels.api as sm
from tensorflow.keras.optimizers import Adam
import joblib

st.markdown("""
## Stock selector for single country equity portfolio
### **App objective**
This app is designed to help you select stocks for a single country equity portfolio.
The Brazil MSCI index is the benchmark for our Brazil portfolio.  After combining ordinary and preference shares, three names account for 39% of the index. 
""")

# Define the DataFrame
br_const = pd.DataFrame({
    'Constituent': ['Petrobras', 'Vale', 'Itau', 'other'],
    'weight': [19.0, 12.1, 8.5, 61.4]
})

# Define labels and sizes for the pie chart
labels = ['Petrobras', 'Vale', 'Itau', 'other']
sizes = [19.0, 12.1, 8.5, 61.4]

# Create the pie chart
fig, ax = plt.subplots(figsize=(0.7,0.7))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', textprops={'fontsize': 3})  
ax.set_title("Brazil MSCI weights", fontsize=5)  # Adjust the font size as needed

# Display the pie chart in Streamlit
st.pyplot(fig)

st.markdown("""
The purpose of this app is to use historical data and backtested relationships to exclude one of the top three names from our portfolio.  This single decision, if correct most of the time, should ensure our portfolio generates positive Alpha.
""")

st.markdown("""
### **Data selection and pre-processing**
The variables we are trying to predict, what we can call the dependent variables, are the ***change*** in the ***relative price*** of **Petrobras to Vale**, **Vale to Itau**, and **Petrobras to Itau**. We use relative price because our focus is on relative performance. We use change to identify the performance over the period.  Our model is built on end-of-month monthly data. We are predicting the performance over the coming two-month period.  

For explanatory variables, we use five Brazil macro series - retail sales, loans, industrial output, consumer confidence and fdi – and three non-Brazilian series -  US retail sales and China loans.  The macro series are seasonally adjusted, 1-month rates of change. We also include the Brazil equity market index, the Bovespa, the S&P 500, the 5-year treasury yield and the Brazil Real to USD rate.

### **Data processing flowchart**
Stock data is downloaded on a total return basis from Bloomberg, code DAY_TO_DAY_TOT_RETURN_GROSS_DVDS daily. Other financial data is PX_Last. Macro data is downloaded monthly, end of month. This daily and monthly data are uploaded to Python separately.

The stock data is first converted to a wealth index, which generates a total return price line. It is then converted to an end-of-month price, and then a monthly change. There is no seasonal adjustment or standardization at this stage. This data in ARIMA terms is (0,1,1).

Some of the macro data is provided as seasonally adjusted change mom. The data that is end of month values is seasonally adjusted and a mom value is calculated. It is then standardized.

The monthly financial data is not seasonally adjusted. We calculate the mon change. Then we standardize. 

Later, in the RandomForest and LSTM models, we further scale and lag the macro data by one period so that it can be used as a predictor of the dependent variables' performance in the following month. 

### **Three analytical approaches**
You can select from three approaches. We can add approaches to our model as desired.

1.	SARIMAX – a multiple regression model with one element of the regression the one-period lag of the period we are trying to predict to capture the momentum effect. 
2.	K nearest neighbors regression (KNN) – a machine learning technique, which seeks to find the best fit in space between the dependent variable and a series of independent variables across a similar number of datapoints (the nearest neighbors). Prior to running KNN, we can run another machine learning technique, RandomForest, which randomly tests our explanatory variables and on this basis selects those offer the greatest explanatory power to be run in the KNN. We choose only variables that score 1 to 3 (on a scale up to 15) to be run in the KNN model.
3.	Long Short-Term Memory (LSTM) – a sequence model, which selects explanatory variables in several stages and passes the results to a subsequent stage that may select different explanatory variables. It’s considered a neural network model because that sequential process mimics our brain function. 

### **How to use**
Select the equity pair for the model results.

This shows the Mean Average Error (lower is better, 0 is a model with perfect forecasting), Root Square Mean Deviation (lower is better, 0 is a model with perfect forecasting), and Mean Absolute Percentage Error (% difference between predicted and actuals).

""")

# Read the statistics from the CSV file
stats_df = pd.read_csv('test_comp_stats.csv')

# Add radio buttons for model selection
selected_model = st.radio('Select the model:', ['PV', 'PI', 'VI'])

# Function to filter and display statistics based on selected model
def display_statistics(model_prefix):
    filtered_stats = stats_df[stats_df['Model'].str.startswith(model_prefix)]
    st.write(f"Model Performance Statistics for {model_prefix}:")
    st.table(filtered_stats.reset_index(drop=True))

# Display statistics based on selected radio button
display_statistics(selected_model)

# SARIMAX model forecast
# Load SARIMAX models
with open('model_PV.pkl', 'rb') as f:
    model_PV = pickle.load(f)
with open('model_PI.pkl', 'rb') as f:
    model_PI = pickle.load(f)
with open('model_VI.pkl', 'rb') as f:
    model_VI = pickle.load(f)

# Define SARIMAX prediction function
def predict_ols(x, model, cols):
    df = pd.DataFrame([x], columns=cols)
    df_with_const = sm.add_constant(df, has_constant='add')
    return model.predict(df_with_const)[0]

# Streamlit app - SARIMAX Section
st.title("Forecast the Next Period")

st.header("SARIMAX Model")

# Checkbox for selecting series
options_sarimax = {
    'PV-SARIMAX': 'PV',
    'PI-SARIMAX': 'PI',
    'VI-SARIMAX': 'VI'
}

selected_option_sarimax = st.radio("Select Series for SARIMAX Model:", list(options_sarimax.keys()))

if selected_option_sarimax:
    series_sarimax = options_sarimax[selected_option_sarimax]

    st.write(f"Selected: {series_sarimax} with SARIMAX")

    if series_sarimax == 'PV':
        x1_input = st.number_input('Lag of PV (PV_lag1):', format="%.3f", step=0.001)
        x2_input = st.number_input('Lag of Bovespa (BOV_ch_lag1):', format="%.3f", step=0.001)
        x3_input = st.number_input('Lag of change of Brazilian loans (br_loans_lag1):', format="%.3f", step=0.001)

        if st.button('Predict PV using SARIMAX'):
            prediction = predict_ols([x1_input, x2_input, x3_input], model_PV, ['PV_lag1', 'BOV_ch_lag1', 'br_loans_lag1'])
            st.write(f"Prediction for next month's PV using SARIMAX: {prediction:.4f}")

    elif series_sarimax == 'PI':
        x1_input = st.number_input('Lag of PI (PI_lag1):', format="%.3f", step=0.001)
        x2_input = st.number_input('Lag of Bovespa (BOV_ch_lag1):', format="%.3f", step=0.001)
        x3_input = st.number_input('Lag of change of Brazilian loans (br_loans_lag1):', format="%.3f", step=0.001)

        if st.button('Predict PI using SARIMAX'):
            prediction = predict_ols([x1_input, x2_input, x3_input], model_PI, ['PI_lag1', 'BOV_ch_lag1', 'br_loans_lag1'])
            st.write(f"Prediction for next month's PI using SARIMAX: {prediction:.4f}")

    elif series_sarimax == 'VI':
        x1_input = st.number_input('Lag of VI (VI_lag1):', format="%.3f", step=0.001)
        x2_input = st.number_input('Lag of Bovespa (BOV_ch_lag1):', format="%.3f", step=0.001)
        x3_input = st.number_input('Lag of change of Brazilian loans (br_loans_lag1):', format="%.3f", step=0.001)

        if st.button('Predict VI using SARIMAX'):
            prediction = predict_ols([x1_input, x2_input, x3_input], model_VI, ['VI_lag1', 'BOV_ch_lag1', 'br_loans_lag1'])
            st.write(f"Prediction for next month's VI using SARIMAX: {prediction:.4f}")


#forecast with KNN
#failed to create because StandardScaler trained on five, kneighborsregressor takes 5. 
#gave up after two hours
            
#Selected features for PV_k: ['PI_lag1' 'VI_lag1' 'BOV_ch_lag1' 'SPY_ch_lag1' 'br_loans_lag1']
#Selected features for PI_k: ['PV_lag1' 'BRL_ch_lag1' 'BOV_ch_lag1' 'SPY_ch_lag1' 'br_loans_lag1']
#Selected features for VI_k: ['PV_lag1' 'PI_lag1' 'SPY_ch_lag1' 'br_loans_lag1' 'us_ret_mom_lag1']

# Load the wealth index data
combined_wealth_PV = pd.read_csv('combined_wealth_PV.csv', index_col=0, parse_dates=True)
combined_wealth_PI = pd.read_csv('combined_wealth_PI.csv', index_col=0, parse_dates=True)
combined_wealth_VI = pd.read_csv('combined_wealth_VI.csv', index_col=0, parse_dates=True)

# Streamlit app
st.title("Wealth Index Visualization")

# Plot for PV
st.subheader('Wealth Index Over Time for PV Strategies')
fig_pv, ax_pv = plt.subplots(figsize=(14, 8))
ax_pv.plot(combined_wealth_PV.index, combined_wealth_PV['Wealth_Index_PV_OLS'], label='Wealth Index PV OLS', color='blue')
ax_pv.plot(combined_wealth_PV.index, combined_wealth_PV['Wealth_Index_PV_KNN'], label='Wealth Index PV KNN', color='purple')
#ax_pv.plot(combined_wealth_PV.index, combined_wealth_PV['Wealth_Index_PV_LSTM'], label='Wealth Index PV LSTM', color='green')
ax_pv.plot(combined_wealth_PV.index, combined_wealth_PV['Wealth_Index_PV_Portfolio'], label='Wealth Index PV Portfolio', color='black')
ax_pv.set_title('Wealth Index Over Time for PV Strategies')
ax_pv.set_xlabel('Date')
ax_pv.set_ylabel('Wealth Index')
ax_pv.legend()
st.pyplot(fig_pv)

# Plot for PI
st.subheader('Wealth Index Over Time for PI Strategies')
fig_pi, ax_pi = plt.subplots(figsize=(14, 8))
ax_pi.plot(combined_wealth_PI.index, combined_wealth_PI['Wealth_Index_PI_OLS'], label='Wealth Index PI OLS', color='blue')
ax_pi.plot(combined_wealth_PI.index, combined_wealth_PI['Wealth_Index_PI_KNN'], label='Wealth Index PI KNN', color='purple')
#ax_pi.plot(combined_wealth_PI.index, combined_wealth_PI['Wealth_Index_PI_LSTM'], label='Wealth Index PI LSTM', color='green')
ax_pi.plot(combined_wealth_PI.index, combined_wealth_PI['Wealth_Index_PI_Portfolio'], label='Wealth Index PI Portfolio', color='black')
ax_pi.set_title('Wealth Index Over Time for PI Strategies')
ax_pi.set_xlabel('Date')
ax_pi.set_ylabel('Wealth Index')
ax_pi.legend()
st.pyplot(fig_pi)

# Plot for VI
st.subheader('Wealth Index Over Time for VI Strategies')
fig_vi, ax_vi = plt.subplots(figsize=(14, 8))
ax_vi.plot(combined_wealth_VI.index, combined_wealth_VI['Wealth_Index_VI_OLS'], label='Wealth Index VI OLS', color='blue')
ax_vi.plot(combined_wealth_VI.index, combined_wealth_VI['Wealth_Index_VI_KNN'], label='Wealth Index VI KNN', color='purple')
#ax_vi.plot(combined_wealth_VI.index, combined_wealth_VI['Wealth_Index_VI_LSTM'], label='Wealth Index VI LSTM', color='green')
ax_vi.plot(combined_wealth_VI.index, combined_wealth_VI['Wealth_Index_VI_Portfolio'], label='Wealth Index VI Portfolio', color='black')
ax_vi.set_title('Wealth Index Over Time for VI Strategies')
ax_vi.set_xlabel('Date')
ax_vi.set_ylabel('Wealth Index')
ax_vi.legend()
st.pyplot(fig_vi)
