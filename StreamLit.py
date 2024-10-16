import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')  # Load the fitted scaler

# Define the min-max values for normalization
min_max_values = {
    'KM': {'min': 0, 'max': 5500000},  # Example values for KM
    'ManufacturingYear': {'min': 1985, 'max': 2023},  # Example values for ManufacturingYear
}

# Normalization function (min-max scaling)
def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value)

# Function to encode categorical variables and normalize numeric variables
def encode_input(KM, ManufacturingYear, OwnerNo, FuelType, City, BodyType, InsuranceValidity, Transmission):
    # Normalize numerical features
    KM_norm = normalize(KM, min_max_values['KM']['min'], min_max_values['KM']['max'])
    ManufacturingYear_norm = normalize(ManufacturingYear, min_max_values['ManufacturingYear']['min'],
                                       min_max_values['ManufacturingYear']['max'])

    # Create a dictionary to store the one-hot encoded categorical features
    data = {
        'KM': KM_norm,
        'ManufacturingYear': ManufacturingYear_norm,
        'OwnerNo': OwnerNo,
        'City_Bangalore': 0, 'City_Chennai': 0, 'City_Delhi': 0, 'City_Hyderabad': 0, 'City_Jaipur': 0, 'City_Kolkata': 0,
        'FuelType_CNG': 0, 'FuelType_Diesel': 0, 'FuelType_Electric': 0, 'FuelType_LPG': 0, 'FuelType_Petrol': 0,
        'BodyType_Convertibles': 0, 'BodyType_Coupe': 0, 'BodyType_Hatchback': 0, 'BodyType_Hybrids': 0,
        'BodyType_MUV': 0, 'BodyType_Minivan': 0, 'BodyType_Minivans': 0, 'BodyType_Pickup Trucks': 0, 'BodyType_SUV': 0,
        'BodyType_Sedan': 0, 'BodyType_Wagon': 0,
        'Transmission_Automatic': 0, 'Transmission_Manual': 0,
        'InsuranceValidity_Comprehensive': 0, 'InsuranceValidity_Third Party': 0, 'InsuranceValidity_Zero Dep': 0
    }

    # Set the corresponding city
    city_column = f'City_{City}'
    if city_column in data:
        data[city_column] = 1

    # Set the corresponding fuel type
    fuel_type_column = f'FuelType_{FuelType}'
    if fuel_type_column in data:
        data[fuel_type_column] = 1

    # Set the corresponding body type
    body_type_column = f'BodyType_{BodyType}'
    if body_type_column in data:
        data[body_type_column] = 1

    # Set the corresponding transmission type
    transmission_column = f'Transmission_{Transmission}'
    if transmission_column in data:
        data[transmission_column] = 1

    # Set the corresponding insurance validity
    insurance_column = f'InsuranceValidity_{InsuranceValidity}'
    if insurance_column in data:
        data[insurance_column] = 1

    # Convert the dictionary to a list of values (order must match the trained model)
    feature_list = [
        data['KM'], data['ManufacturingYear'], data['OwnerNo'],
        data['City_Bangalore'], data['City_Chennai'], data['City_Delhi'], data['City_Hyderabad'], data['City_Jaipur'],
        data['City_Kolkata'],
        data['FuelType_CNG'], data['FuelType_Diesel'], data['FuelType_Electric'], data['FuelType_LPG'], data['FuelType_Petrol'],
        data['BodyType_Convertibles'], data['BodyType_Coupe'], data['BodyType_Hatchback'], data['BodyType_Hybrids'],
        data['BodyType_MUV'], data['BodyType_Minivan'], data['BodyType_Minivans'], data['BodyType_Pickup Trucks'],
        data['BodyType_SUV'], data['BodyType_Sedan'], data['BodyType_Wagon'],
        data['Transmission_Automatic'], data['Transmission_Manual'],
        data['InsuranceValidity_Comprehensive'], data['InsuranceValidity_Third Party'], data['InsuranceValidity_Zero Dep']
    ]

    return feature_list

# Streamlit app setup
st.title("Car Price Prediction")

# User inputs
KM = st.number_input("Kilometers driven", min_value=0, max_value=1000000, step=100)
ManufacturingYear = st.number_input("Manufacturing Year", min_value=1990, max_value=2024)
OwnerNo = st.number_input("Number of Previous Owners", min_value=0, max_value=10)

FuelType = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "Electric", "LPG"])
City = st.selectbox("City", ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Jaipur", "Kolkata"])
BodyType = st.selectbox("Body Type",
                        ["Convertibles", "Coupe", "Hatchback", "Hybrids", "MUV", "Minivan", "Minivans", "Pickup Trucks",
                         "SUV", "Sedan", "Wagon"])

InsuranceValidity = st.selectbox("Insurance Validity", ['Third Party', 'Zero Dep', 'Comprehensive'])
Transmission = st.selectbox("Transmission", ['Manual', 'Automatic'])


# Button for prediction
if st.button("Predict Price"):
    # Prepare input data
    input_data = np.array([encode_input(KM, ManufacturingYear, OwnerNo, FuelType, City, BodyType,
                                        InsuranceValidity, Transmission)])

    # Make prediction (normalized)
    predicted_price_normalized = model.predict(input_data)

    # Denormalize the predicted price using the loaded scaler
    predicted_price_normalized = np.array(predicted_price_normalized).reshape(-1, 1)  # Reshape to 2D array
    predicted_price = scaler.inverse_transform(predicted_price_normalized)

    # Display the denormalized predicted price
    st.success(f"The predicted price is {predicted_price[0][0]:.2f} lakhs")
