import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import joblib

# Load the  dataset
df = pd.read_excel("Clean_file.xlsx")

features = ['KM','ManufacturingYear','OwnerNo','City_Bangalore','City_Chennai','City_Delhi','City_Hyderabad',
 'City_Jaipur','City_Kolkata','FuelType_CNG','FuelType_Diesel','FuelType_Electric', 'FuelType_LPG',
 'FuelType_LPG','FuelType_Petrol','BodyType_Convertibles','BodyType_Coupe','BodyType_Hatchback','BodyType_Hybrids',
 'BodyType_MUV','BodyType_Minivan','BodyType_Minivans','BodyType_Pickup Trucks','BodyType_SUV','BodyType_Sedan',
 'BodyType_Wagon','Transmission_Automatic','Transmission_Manual','InsuranceValidity_Comprehensive','InsuranceValidity_Third Party','InsuranceValidity_Zero Dep'
]

scaler = MinMaxScaler()
df['Numeric_Price'] = scaler.fit_transform(df[['Numeric_Price']])
# Extract X (features) from the dataframe
X = df[features]
y = df['Numeric_Price']  # Target is the car price

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest Regressor
clf = RandomForestRegressor(n_estimators=3, random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluating regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


"""	Feature Engineering: Creating new features to improve model performance """

#'df' is our data with categorical and numerical features
categorical_features =  ['City_Bangalore','City_Chennai','City_Delhi','City_Hyderabad',
 'City_Jaipur','City_Kolkata','FuelType_CNG','FuelType_Diesel','FuelType_Electric', 'FuelType_LPG',
 'FuelType_LPG','FuelType_Petrol','BodyType_Convertibles','BodyType_Coupe','BodyType_Hatchback','BodyType_Hybrids',
 'BodyType_MUV','BodyType_Minivan','BodyType_Minivans','BodyType_Pickup Trucks','BodyType_SUV','BodyType_Sedan',
 'BodyType_Wagon','Transmission_Automatic','Transmission_Manual','InsuranceValidity_Comprehensive','InsuranceValidity_Third Party','InsuranceValidity_Zero Dep'
]

numerical_features = ['ManufacturingYear','OwnerNo']

#  Create new features based on domain knowledge
df['Car_Age'] = 2024 - df['ManufacturingYear']  # Creating new feature 'Car Age' from existing feature 'ManufacturingYear'
df['Mileage_per_Year'] = df['KM'] / df['Car_Age']  # Creating new feature 'Mileage per Year'

# Select all the features for the model not include new feature because the performance of model will remain constant after adding new features also so it is not necessary to add them
all_features = numerical_features  + categorical_features

# Extract features and target
X = df[all_features]
y = df['Numeric_Price']  # Target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestRegressor(n_estimators=300, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Evaluating regression metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")

""" Hyperparameter tuning with cross-validation to ensure robust performance"""

# Define the hyperparameter grid for tuning
param_grid = {
    'n_estimators': [100, 200, 300],            # Number of trees
    'max_depth': [10, 15, 20],                  # Depth of each tree
    'min_samples_split': [2, 5, 10],            # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],              # Minimum samples at a leaf node
    'max_features': [None, 'sqrt', 'log2']    # Number of features to consider at each split
}

# Set up GridSearchCV for hyperparameter tuning with cross-validation
grid_search = GridSearchCV(estimator=RandomForestRegressor(random_state=42),
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           scoring='neg_mean_squared_error',
                           n_jobs=-1,  # Use all available cores
                           verbose=2)

# Fit the model with the best hyperparameters
grid_search.fit(X_train, y_train)

# Best hyperparameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Print best hyperparameters and MSE from cross-validation
print("Best Hyperparameters: ", best_params)
print("Best Cross-Validation MSE: ", -best_score)

# Fit the final model with the best hyperparameters
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train)

# Evaluate the final model on the test set
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Print the metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"R-squared (R2): {r2:.2f}")


# Save the final model
joblib.dump(best_model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')