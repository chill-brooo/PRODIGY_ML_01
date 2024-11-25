import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import zipfile
import os

# Step 1: Print the current working directory
print("Current Working Directory:", os.getcwd())

# Define the path to the ZIP file
zip_file_path = r'C:\Users\Ved Thombre\Python Download\Python codes\house-prices-advanced-regression-techniques.zip'

# Define the directory where you want to extract the files
extraction_directory = r'C:\Users\Ved Thombre\Python Download\Python codes'

# Step 4: Check if the ZIP file exists
if not os.path.exists(zip_file_path):
    print(f"The specified ZIP file was not found: {zip_file_path}")
else:
    # Step 5: Extract the ZIP file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extraction_directory)

    # Step 6: List the files in the extraction directory
    print("Extracted files:", os.listdir(extraction_directory))

    # Step 7: Load the dataset
    try:
        train_data = pd.read_csv(os.path.join(extraction_directory, 'train.csv'))  # Load training data
    except FileNotFoundError:
        print("The specified file 'train.csv' was not found. Please check the file path.")
        raise  # Stop execution if the file is not found

    # Display the first few rows of the dataset
    print("Training Data Sample:")
    print(train_data.head())

    # Step 8: Check for required columns
    required_columns = ['square_footage', 'bedrooms', 'bathrooms', 'price']
    missing_columns = [col for col in required_columns if col not in train_data.columns]

    if missing_columns:
        print(f"Warning: The following expected columns are missing: {missing_columns}")
    else:
        # Define features and target variable
        X = train_data[['square_footage', 'bedrooms', 'bathrooms']]  # Features
        y = train_data['price']  # Target variable

        # Step 9: Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Step 10: Create a Linear Regression model
        model = LinearRegression()

        # Step 11: Train the model
        model.fit(X_train, y_train)

        # Step 12: Make predictions
        y_pred = model.predict(X_test)

        # Step 13: Calculate Mean Squared Error and R-squared
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Output the performance metrics
        print(f'Mean Squared Error: {mse}')
        print(f'R² Score: {r2}')

        # Step 14: Visualize Actual vs Predicted prices
        plt.scatter(y_test, y_pred)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title('Actual vs Predicted Prices')
        plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Diagonal line
        plt.show()