# Used Car Price Prediction:

   This project aims to predict the price of used cars based on various features like mileage, year of manufacture, engine specifications, and other attributes. A    Random Forest Regressor is used as the final model after a series of preprocessing, exploratory data analysis, and model evaluation steps. The project also includes    the development of an interactive Streamlit application for real-time car price predictions.
   
# Prerequisites

   ->Python 3.x: Required to run the project.
   ->Python Libraries: Install the dependencies via pip.

# Libraries Used:

 *pandas: Data manipulation and analysis
 *numpy: Numerical operations
 *scikit-learn:
    ->RandomForestRegressor: The final model used for price prediction
    ->train_test_split: For splitting the dataset into training and testing sets
    ->GridSearchCV: For hyperparameter tuning
    ->mean_absolute_error, mean_squared_error, r2_score: For model evaluation
    ->MinMaxScaler: For normalizing numeric features
 *joblib: To save and load models
 *seaborn: For data visualization
 *matplotlib: For plotting graphs
 *scipy:
    ->stats: For statistical analysis
    ->skew, kurtosis: For distribution analysis
 *Streamlit: To build an interactive web application for price prediction

# Project Steps

 Step 1: Data Structuring: Clean, structure, and merge raw data from different cities into a single dataset, using file Data_Processed.py

 Step 2: Data Preprocessing, This step is performed using Encoding_Normalization.py:

        * Handle missing values using imputatio(Mean, Median, Mode).
        * Standardize data formats and encode categorical features (One-Hot Encoding).
        * Remove outliers using IQR.
        * Normalize numeric features with MinMaxScaler.

 Step 3: EDA:
        * Feature selection using correlation analysis.
        * Calculate statistics (mean, median, skewness, kurtosis) and visualize data.

 Step 4: Model Development, This step is prformed using Normalizing_Model.py:
        * Split data (80-20)train-test set and train models (Linear Regression, Decision Trees, Random Forest).
        * Evaluate models with metrics (MAE, MSE, RMSE, R²).
        * Use GridSearchCV for hyperparameter tuning and select the best model (Random Forest).

 Step 5: Model Saving: Save the best model using joblib.

 Step 6: Streamlit Application: Build an app for users to input car details (e.g., KM, year) and get real-time price predictions.
	 This step is performed using StreamLit.py