import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium


# Load the train.parquet file
df_train = pd.read_parquet("./Dataset_parquet/train.parquet")

# Convert the dataframe to a CSV file
df_train.to_csv("./Dataset_csv/train.csv", index=False)

# Load the test.parquet file
df_test = pd.read_parquet("./Dataset_parquet/test.parquet")

# Convert the dataframe to a CSV file
df_test.to_csv("./Dataset_csv/test.csv", index=False)

# Load the train file
df = pd.read_csv("./Dataset_csv/train.csv")

# Check for missing values
print(df.isnull().sum())

'''# Create a function to assign the price scale based on the price
def assign_price_scale(price):
    if price >= 0 and price <= 999:
        return 'low'
    elif price >= 1000 and price <= 1999:
        return 'medium'
    elif price >= 2000:
        return 'high'

# Apply the function to the 'price' column and create a new column 'price_scale'
df['price_scale'] = df['price'].apply(lambda x: assign_price_scale(x))

# Print the first few rows of the dataframe to check if the new column was created correctly
print(df.head())'''

# MODIFICATION: 
#[https://soyhenry.slack.com/archives/C03UYCDH8NN/p1674509452835809](https://soyhenry.slack.com/archives/C03UYCDH8NN/p1674509452835809)

#now the scale for "price_scale" colum is in the range of low-notlow
### NEW RETURN: low = 0, notlow = 1

# Create a function to assign the price scale based on the price
def assign_price_scale(price):
    if price >= 0 and price <= 999:
        return 1 # low
    else:
        return 0 # not low

# Apply the function to the 'price' column and create a new column 'price_scale'
df['price_scale'] = df['price'].apply(lambda x: assign_price_scale(x))

# More exploration:
## Replacing missing values

# Check unique values in 'laundry_options' column
print("Unique values in 'laundry_options' column:", df['laundry_options'].unique())

# Check unique values in 'parking_options' column
print("Unique values in 'parking_options' column:", df['parking_options'].unique())

# Strategy:
## Replace null in the "laundry_options" and "parking_options" for "unknown" value for further analisys

# Replace null values in 'laundry_options' column with 'unknown'
df['laundry_options'] = df['laundry_options'].fillna('unknown')

# Replace null values in 'parking_options' column with 'unknown'
df['parking_options'] = df['parking_options'].fillna('unknown')

# Check

# Check if there are any remaining null values in the 'laundry_options' column
print(df['laundry_options'].isnull().sum())

# Check if there are any remaining null values in the 'parking_options' column
print(df['parking_options'].isnull().sum())

# Display rows without gps cordinates

# Get the rows with null values in 'lat' and 'long' columns
null_lat_long_df = df.loc[(df['lat'].isnull()) & (df['long'].isnull())]

# Print the dataframe
print(null_lat_long_df)

# Display map usin folium to check the scope

# Lat and Long variables

latitude = df['lat'].mean()
longitude = df['long'].mean()
m = folium.Map(location=[latitude, longitude], zoom_start=13)

# Create a map object
m = folium.Map(location=[latitude, longitude], zoom_start=13)

# Add a marker to the map
folium.Marker(location=[latitude, longitude]).add_to(m)

# Display the map
m

# Due to performance problems
# I had to che the map ploting in a diferent file

# Creation of a new dataframe just for id, lat and long

cord_df = df[['id', 'lat', 'long']]

# Display head and tail for check

# Display the first 5 rows of the dataframe
print(cord_df.head())

# Display the last 5 rows of the dataframe
print(cord_df.tail())

# Exportin cord_df as a .csv file to che it in the new .py file

cord_df.to_csv('./Map_check/train_only_cordinates.csv', index=False)

# Display df types
print(df.info())

# Analisys number of duplicates per colum

# First chek all rows
row_count = df.shape[0]
print(row_count)

# Duplicates in "image_url"
duplicates_mask = df['image_url'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate image_url:", duplicates_count)

# Duplicates in "region_url"
duplicates_mask = df['region_url'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate region_url:", duplicates_count)

# Duplicates in "url"
duplicates_mask = df['url'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate url:", duplicates_count)

# Duplicates in "description"
duplicates_mask = df['description'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate description:", duplicates_count)

# Duplicates in "lat"
duplicates_mask = df['lat'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate lat:", duplicates_count)

# Duplicates in "long"
duplicates_mask = df['long'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate long:", duplicates_count)

# Drop duplicate rows fitering by image_url
df = df.drop_duplicates(subset=['image_url'], keep='first')

# Rechek all rows
row_count = df.shape[0]
print(row_count)

# Duplicates in "image_url"
duplicates_mask = df['image_url'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate image_url:", duplicates_count)

# Duplicates in "region_url"
duplicates_mask = df['region_url'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate region_url:", duplicates_count)

# Duplicates in "url"
duplicates_mask = df['url'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate url:", duplicates_count)

# Duplicates in "description"
duplicates_mask = df['description'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate description:", duplicates_count)

# Duplicates in "lat"
duplicates_mask = df['lat'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate lat:", duplicates_count)

# Duplicates in "long"
duplicates_mask = df['long'].duplicated()
duplicates_count = duplicates_mask.sum()
print("Number of duplicate long:", duplicates_count)

# Print the list of columns
print(df.columns)

# Iterate over each column in the DataFrame
for col in df.columns:
    # Print the unique values and type for the current column
    print("Unique values in '{}' column: {}".format(col, df[col].unique()))
    print("Type of '{}' column: {}".format(col, df[col].dtype))

# Replace the Null values in "lat" & "long" colums with general cordinates using the "satate" colum

# Create a dictionary to map state abbreviation to coordinates
state_coords = {
    'id': (43.6137, -116.2034),
    'co': (39.5501, -105.7821),
    'va': (37.7693, -78.1700),
    'az': (34.2744, -111.6602),
    'tn': (35.7478, -86.6923),
    'tx': (31.9686, -99.9018),
    'ma': (42.4072, -71.3824),
    'ks': (38.5111, -98.2883),
    'pa': (40.5773, -77.264),
    'me': (45.2538, -69.4455),
    'ms': (32.7416, -89.6787),
    'la': (31.1801, -91.8749),
    'mn': (46.729553, -94.6859),
    'ky': (37.669, -84.6514),
    'wa': (47.7511, -120.7401),
    'al': (32.806671, -86.791130),
    'ca': (37.271875, -119.270415),
    'ga': (32.678125, -83.223905),
    'nc': (35.787743, -78.634106),
    'md': (39.045755, -76.641271),
    'ia': (41.9383, -93.0977),
    'fl': (27.994402, -81.760254),
    'or': (43.804133, -120.554201),
    'ar': (34.799999, -92.199997),
    'hi': (19.8968, -155.5828),
    'nj': (40.143006, -74.731116),
    'ut': (39.419220, -111.950684),
    'wv': (38.597626, -80.454903),
    'ny': (42.165726, -74.948051),
    'sc': (33.836081, -81.1637245),
    'oh': (40.417287, -82.907123),
    'mo': (38.573936, -92.60376),
    'mi': (44.314844, -85.602364),
    'ak': (61.3850, -152.2683),
    'sd': (44.2126, -99.4388),
    'ok': (35.5376, -96.9000)
}

# Get the rows where 'lat' is null
null_lat_rows = df[df['lat'].isnull()]

# Iterate over the rows where 'lat' is null
for index, row in null_lat_rows.iterrows():
    # Get the state abbreviation from the 'state' column
    state_abbr = row['state']
    if state_abbr in state_coords:
        # Look up the coordinates for the state abbreviation in the dictionary
        lat = state_coords[state_abbr][0]
        # Update the 'lat' column for the current row
        df.loc[index, 'lat'] = lat

# Get the rows where 'long' is null
null_long_rows = df[df['long'].isnull()]

# Iterate over the rows where 'long' is null
for index, row in null_long_rows.iterrows():
    # Get the state abbreviation from the 'state' column
    state_abbr = row['state']
    if state_abbr in state_coords:
        # Look up the coordinates for the state abbreviation in the dictionary
        long = state_coords[state_abbr][1]
        # Update the 'long' column for the current row
        df.loc[index, 'long'] = long

# Check for missing values
print(df.isnull().sum())

# Create a boolean mask to select the rows that meet the conditions
mask = (df['lat'] > 55) | (df['lat'] < 20) | (df['long'] > -55) | (df['long'] < -130) | ~df['state'].isin(state_coords.keys())

# Use the mask to select the rows to drop
df_to_drop = df.loc[mask]

# Use the drop method to remove the selected rows from the DataFrame
df = df.drop(df_to_drop.index)

# Check for missing values
print(df.isnull().sum())

# Display df info
df.info()

# Drop missing values in "description"
df = df.dropna(subset=['description'])

# Display info
df.info()

# Check for missing values
print(df.isnull().sum())

# Heatmap
# Create a heatmap of the DataFrame
sns.heatmap(df.corr(), annot=True)

# Categorical colums into numericals
df['url_encoded'] = pd.factorize(df['url'])[0]
df['region_encoded'] = pd.factorize(df['region'])[0]
df['region_url_encoded'] = pd.factorize(df['region_url'])[0]
df['type_encoded'] = pd.factorize(df['type'])[0]
df['laundry_options_encoded'] = pd.factorize(df['laundry_options'])[0]
df['parking_options_encoded'] = pd.factorize(df['parking_options'])[0]
df['image_url_encoded'] = pd.factorize(df['image_url'])[0]
df['state_encoded'] = pd.factorize(df['state'])[0]

# Anothe heatmap
sns.heatmap(df.corr(), annot=True, annot_kws={"size": 5}, fmt='.2f')

# Correlation matrix for "price_scale" column
corr_matrix = df.corr()
print(corr_matrix['price_scale'])

# Drop unusefull colums
df = df.drop(columns=['id', 'state', 'parking_options', 'laundry_options', 'region', 'type', 'url_encoded', 'url', 'region_url_encoded', 'region_url', 'price', 'image_url_encoded', 'image_url', 'description'])

# Display df info
df.info()

# Save cleaned df as .csv
df.to_csv('./clean_files/train_clean.csv', index=False)

# Importing test.csv for normalization
df_test = pd.read_csv('./Dataset_csv/test.csv')

# Categorical colums into numericals for df_test
df_test['url_encoded'] = pd.factorize(df_test['url'])[0]
df_test['region_encoded'] = pd.factorize(df_test['region'])[0]
df_test['region_url_encoded'] = pd.factorize(df_test['region_url'])[0]
df_test['type_encoded'] = pd.factorize(df_test['type'])[0]
df_test['laundry_options_encoded'] = pd.factorize(df_test['laundry_options'])[0]
df_test['parking_options_encoded'] = pd.factorize(df_test['parking_options'])[0]
df_test['image_url_encoded'] = pd.factorize(df_test['image_url'])[0]
df_test['state_encoded'] = pd.factorize(df_test['state'])[0]

# Drop unusefull colums for df_test
df_test = df_test.drop(columns=['id', 'state', 'parking_options', 'laundry_options', 'region', 'type', 'url_encoded', 'url', 'region_url_encoded', 'region_url', 'image_url_encoded', 'image_url', 'description'])

# Save cleaned df_test as .csv
df_test.to_csv('./Clean_files/test_clean.csv', index=False)