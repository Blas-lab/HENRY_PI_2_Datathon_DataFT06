# Import the necessary libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Explore features options:

# Features:
## ['sqfeet', 'beds', 'baths', 'cats_allowed', 'dogs_allowed', 'smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished', 'lat', 'long', 'region_encoded', 'type_encoded', 'laundry_options_encoded', 'parking_options_encoded', 'state_encoded']
## Excluded: none

# Load the data into a DataFrame
df = pd.read_csv('./Clean_files/train_clean.csv')

# Features Creation:

## Feature new colum "pet_allowed" as a mean between "cats_allowed" & "dogs_allowed"
df['pets_allowed'] = df[['cats_allowed', 'dogs_allowed']].mean(axis=1)

# Drop the original columns
df = df.drop(['cats_allowed', 'dogs_allowed'], axis=1)

# Outlier normalization

# Define the columns to check for outliers
cols_to_check = ['sqfeet', 'beds', 'baths', 'region_encoded', 'type_encoded', 'parking_options_encoded', 'state_encoded']

# Define the threshold for outliers
threshold = 3

# Iterate through the columns and remove outliers
for col in cols_to_check:
    mean = df[col].mean()
    std = df[col].std()
    df = df[(df[col] < mean + threshold*std) & (df[col] > mean - threshold*std)]

# Reset the index
df.reset_index(inplace=True, drop=True)

# Model Training

# Features:
## Included: ['sqfeet', 'beds', 'baths', 'pets_allowed', 'smoking_allowed', 'electric_vehicle_charge', 'comes_furnished', 'lat', 'long', 'region_encoded', 'type_encoded', 'laundry_options_encoded', 'parking_options_encoded', 'state_encoded']
## Excluded:[ 'wheelchair_access',]

# Define the features and target variables
cols_to_use = ['sqfeet', 'beds', 'baths', 'pets_allowed', 'smoking_allowed', 'laundry_options_encoded', 'electric_vehicle_charge', 'comes_furnished', 'region_encoded', 'type_encoded', 'laundry_options_encoded', 'parking_options_encoded', 'state_encoded']
X = df[cols_to_use]
y = df['price_scale']


# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.42, random_state=42)

# Create a Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)

# Fit the model to the training data
dt.fit(X_train, y_train)

# Make predictions on the test set
y_pred = dt.predict(X_test)

# Calculate the mean absolute error of the model
mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error:", mae)

# Use the model to make predictions on the entire dataset
df_predictions = df.copy()
df_predictions['price_scale_predicted'] = dt.predict(df[cols_to_use])
df_predictions.to_csv('./Predictions_sup/prediction_dt.csv', index=False)

# Check df
print(df)

# Model Testing with "test_clean.csv"

# Load the test data into a DataFrame
df_test = pd.read_csv('./Clean_files/test_clean.csv')

# Use the same columns that were used for training
X_test_data = df_test[cols_to_use]

# import SimpleImputer class
from sklearn.impute import SimpleImputer

# Create an imputer object
imputer = SimpleImputer(strategy='mean')

# Fit the imputer to the training data
imputer.fit(X_train)

# Transform the test data using the imputer
X_test_imputed = imputer.transform(X_test_data)

# Make predictions on the test data
y_pred_test = dt.predict(X_test_imputed)

# Add the predicted values to the test data DataFrame
df_test['price_scale_predicted'] = y_pred_test

# Export the predictions to a CSV file
df_test.to_csv('./Predictions_sup/test_dtc_prediction.csv', index=False)

# Normalization for project presentation
## 1 colum only numerical values "1"&"0"

# Load the predictions from test_dtc_prediction.csv into a DataFrame
df_test_predictions = pd.read_csv('./Predictions_sup/test_dtc_prediction.csv')

# Create a new DataFrame with only the 'price_scale_predicted' column
df_final_format = df_test_predictions[['price_scale_predicted']]

# Export the final format DataFrame to a CSV file
df_final_format.to_csv('./Predictions_sup/final_format.csv', index=False, header=False)
