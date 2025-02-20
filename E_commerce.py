import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
# Ensure sklearmn is installed
# Load dataset
ecom = pd.read_csv('Test.csv', na_values=['', 'NA', 'null'])

# Display basic info
ecom.info()
print(ecom.head())
# Filling the missing Values
## Fill categorical missing values with mode
categorical_cols = ['Date', 'Category', 'Brand']
for cols in categorical_cols:
    ecom[cols].fillna(ecom[cols].mode()[0], inplace=True)

## Fill numerical missing values with mean
numerical_cols = ['Day_of_Week', 'Holiday_Indicator','Past_Purcahse_Trends', 'Price', 'Discount','Competitor_Price']
for cols in numerical_cols:
    ecom[cols].fillna(ecom[cols].mean(), inplace=True)

# Check Sales_Quantity columns
if ecom['Sales_Quantity'].isna().sum() == len(ecom):
    print('Sales_Quantity has no values,  setting it to 0 for now.')
    ecom['Sales_Quantity'] = 0 # Placeholder since no data is available

# Encode categorical features
ecom = pd.get_dummies(ecom, columns=['Category', 'Brand'], drop_first=True) 
#Define features and target
X = ecom.drop(columns=['Sales_Quantity', 'Date']) # Drop target and date
y = ecom['Sales_Quantity']
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)
# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# predict on new data
predictions = model.predict(X_test[:5]) # Predidict first five rows
print(" Predictions for first 5 test samples:", predictions)

