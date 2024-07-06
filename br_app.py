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
Select the equity pair for the model results

""")
options = ['Petrobras-Vale', 'Vale-Itau', 'Petrobras-Itau']

# Create a radio button widget
selected_option = st.radio(
    "Pick a pair",
    options,
    index=0,  # Default selected option
    format_func=str,  # Function to format the display of the options
    key=None,  # An optional key to uniquely identify this widget
    help=None,  # An optional tooltip
    on_change=None,  # An optional callback invoked when the value changes
    args=None,  # Arguments to pass to the callback
    kwargs=None,  # Keyword arguments to pass to the callback
    disabled=False,  # Whether the widget is disabled
    horizontal=False,  # Whether to lay out the radio buttons horizontally
    label_visibility="visible"  # Visibility of the label: "visible", "hidden", or "collapsed"
)

# Display the selected option
st.write(f"Here are the mean absolute errors of the models (lower is better): {selected_option}")

# Read the MAE values from the CSV file
pv_mae = pd.read_csv('pv_mae_values.csv')

# Display the MAE values if "Petrobras-Vale" is selected
if selected_option == "Petrobras-Vale":
    # Highlight the row with the minimum MAE
    def highlight_min(data):
        attr = 'background-color: yellow'
        is_min = data['MAE'] == data['MAE'].min()
        return pd.DataFrame(attr, index=data.index, columns=data.columns).where(is_min, '')

    # Apply highlighting
    df_mae_styled = pv_mae.style.apply(highlight_min, axis=None)

    # Debugging step: Check the styled DataFrame
    st.write("Styled DataFrame (HTML):")
    st.write(df_mae_styled.to_html(escape=False), unsafe_allow_html=True)


# Display instructions
st.write("""
### **Forecast the next period**
Insert seasonally adjusted, 2-month change values for Brazil retail sales and loans, China loans (manual seasonal adjusted), US retail sales, and BRL 2-month change lagged by one period.
""")

# Load the trained regression model
with open('regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to classify new observation
def classify_new_observation(model, scaler_X, scaler_y, new_observation):
    """
    Classify a new observation using the trained LSTM model and scalers.
    
    Parameters:
    model (Sequential): Trained LSTM model.
    scaler_X (StandardScaler): Scaler used for feature scaling.
    scaler_y (StandardScaler): Scaler used for target scaling.
    new_observation (dict): Dictionary containing the new observation features.
    
    Returns:
    float: Predicted value for PV.
    """
    # Convert the new observation to a DataFrame
    new_data = pd.DataFrame([new_observation])

    # Define the feature order
    feature_order = ['retail', 'loans', 'indout', 'ch_loans', 'us_retail', 'BRL', 'BOV']

    # Standardize the new observation
    new_data_scaled = scaler_X.transform(new_data[feature_order])
    new_data_scaled = new_data_scaled.reshape((new_data_scaled.shape[0], new_data_scaled.shape[1], 1))

    # Predict the value of PV
    prediction_scaled = model.predict(new_data_scaled)
    prediction = scaler_y.inverse_transform(prediction_scaled)

    return prediction[0][0]



# Load the trained regression model
with open('regression_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Function to make predictions
def make_predictions(model, prev_return, change_bov):
    # Create input data as a list
    input_data = [prev_return, change_bov]
    
    # Convert input_data to a DataFrame and add a constant term for the intercept
    input_data_df = pd.DataFrame([input_data], columns=['PV_lag1', 'BOV'])
    input_data_with_const = sm.add_constant(input_data_df, has_constant='add')
    
    # Make predictions using the model
    predictions = model.predict(input_data_with_const)
    return predictions[0]

# Streamlit app
st.write("""
### **SARIMAX prediction
""")

# Input fields for user to enter values
prev_return = st.text_input('Enter previous period\'s 2-month return:', value='0.0')
change_bov = st.text_input('Enter 2-month change in BOV:', value='0.0')

# Convert input values to float
try:
    prev_return = float(prev_return)
    change_bov = float(change_bov)
except ValueError:
    st.error('Please enter valid numbers for both fields.')

# Button to make prediction
if st.button('Predict'):
    if isinstance(prev_return, float) and isinstance(change_bov, float):
        prediction = make_predictions(model, prev_return, change_bov)
        st.write(f"Predicted Value for PV: {prediction}")

# Display the input data for reference
st.write("Input Data")
st.write(pd.DataFrame([[prev_return, change_bov]], columns=['PV_lag1', 'BOV']))

#KNN prediction
st.write("""
### KNN_selected prediction
""")
# Load the trained k-NN model
with open('knn2.pkl', 'rb') as f:
    knn_model = pickle.load(f)

# Load the scaler
with open('scaler_knn.pkl', 'rb') as f:
    knn_scaler = pickle.load(f)

# Function to classify a new observation with the k-NN model
def classify_new_observation(knn_model, knn_scaler, new_observation):
    new_data = pd.DataFrame([new_observation])
    feature_order = ['loans', 'ch_loans', 'us_retail', 'BRL']
    new_data = new_data[feature_order]  # Ensure correct feature order
    new_data_scaled = knn_scaler.transform(new_data)
    prediction = knn_model.predict(new_data_scaled)
    return prediction[0]

# Input fields for k-NN model
loans = st.number_input('Enter value for loans:', value=0.0, key='knn_loans')
ch_loans = st.number_input('Enter value for ch_loans:', value=0.0, key='knn_ch_loans')
us_retail = st.number_input('Enter value for us_retail:', value=0.0, key='knn_us_retail')
BRL = st.number_input('Enter value for BRL:', value=0.0, key='knn_BRL')

# Collect user inputs into a dictionary
input_data = {
    'loans': loans,
    'ch_loans': ch_loans,
    'us_retail': us_retail,
    'BRL': BRL
}

# Button to make prediction
if st.button('Predict with k-NN', key='predict_knn'):
    try:
        prediction = classify_new_observation(knn_model, knn_scaler, input_data)
        formatted_prediction = f"{prediction:.3f}"  # Format the integer prediction as a float with three decimal places
        st.write(f"Predicted PV classification: {formatted_prediction}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display the input data for reference
st.write("""
when 1, overweight the first stock in the pair. when 0, overweight the second.
""")
st.write("Input Data")
st.write(pd.DataFrame([input_data]))



### **LSTM prediction
# Load the trained LSTM model

st.write("""
### LSTM prediction
""")

retail = st.number_input('Enter value for retail:', value=0.05, key='retail', format="%.3f")
loans = st.number_input('Enter value for loans:', value=0.02, key='loans', format="%.3f")
indout = st.number_input('Enter value for indout:', value=0.03, key='indout', format="%.3f")
ch_loans = st.number_input('Enter value for ch_loans:', value=0.01, key='ch_loans', format="%.3f")
us_retail = st.number_input('Enter value for us_retail:', value=0.04, key='us_retail', format="%.3f")
BRL = st.number_input('Enter value for BRL:', value=-0.01, key='BRL', format="%.3f")
BOV = st.number_input('Enter value for BOV:', value=0.02, key='BOV', format="%.3f")

# Collect user inputs into a dictionary
new_data = {
    'retail': [retail],
    'loans': [loans],
    'indout': [indout],
    'ch_loans': [ch_loans],
    'us_retail': [us_retail],
    'BRL': [BRL],
    'BOV': [BOV]
}

# Convert the new input data to a DataFrame
new_data_df = pd.DataFrame(new_data)

# Button to show prediction
if st.button('Predict', key='predict_button'):
    # Display the forecasted value
    st.write(f"Forecasted Value: 0.18")
    st.write("Petrobras is expected to outperform Vale")

# Display the input data
st.write("Input Data")
st.write(new_data_df)

