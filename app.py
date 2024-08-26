import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
file_path = 'Housing.csv'
df = pd.read_csv(file_path)

# Preprocess the data
label_encoder = LabelEncoder()
categorical_columns = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                       'airconditioning', 'prefarea', 'furnishingstatus']

# Apply Label Encoding to categorical columns
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# Extract features (X) and target (Y)
X = df.drop(columns=['price'])
Y = df['price'].values.reshape(-1, 1)

# Normalize the numerical features
X = (X - X.mean()) / X.std()
X = X.values


# Initialize weights and bias for the Linear Regression model
def initialize_parameters(n):
    W = np.zeros((n, 1))
    b = 0
    return W, b


# Model Prediction (Hypothesis Function)
def predict(X, W, b):
    return np.dot(X, W) + b


# Gradient Descent for updating weights and bias
def gradient_descent(X, Y, Y_pred, W, b, learning_rate):
    m = len(Y)
    dW = (1 / m) * np.dot(X.T, (Y_pred - Y))
    db = (1 / m) * np.sum(Y_pred - Y)

    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b


# Train the model
def train(X, Y, learning_rate=0.01, iterations=1000):
    W, b = initialize_parameters(X.shape[1])
    for i in range(iterations):
        Y_pred = predict(X, W, b)
        W, b = gradient_descent(X, Y, Y_pred, W, b, learning_rate)
    return W, b


# Train the linear regression model
W, b = train(X, Y, learning_rate=0.01, iterations=1000)

# Streamlit interface
st.title("House Price Prediction")

# User input for house features
area = st.number_input("Area of the house (sq. ft.):", min_value=0)
bedrooms = st.number_input("Number of bedrooms:", min_value=0)
bathrooms = st.number_input("Number of bathrooms:", min_value=0)
stories = st.number_input("Number of stories:", min_value=0)
mainroad = st.selectbox("Is the house on the main road?", ['yes', 'no'])
guestroom = st.selectbox("Does the house have a guest room?", ['yes', 'no'])
basement = st.selectbox("Does the house have a basement?", ['yes', 'no'])
hotwaterheating = st.selectbox("Does the house have hot water heating?", ['yes', 'no'])
airconditioning = st.selectbox("Does the house have air conditioning?", ['yes', 'no'])
parking = st.number_input("Number of parking spaces:", min_value=0)
prefarea = st.selectbox("Is the house in a preferred area?", ['yes', 'no'])
furnishingstatus = st.selectbox("Furnishing status:", ['furnished', 'semi-furnished', 'unfurnished'])

# Encode the categorical inputs
input_data = np.array([[area, bedrooms, bathrooms, stories,
                        1 if mainroad == 'yes' else 0,
                        1 if guestroom == 'yes' else 0,
                        1 if basement == 'yes' else 0,
                        1 if hotwaterheating == 'yes' else 0,
                        1 if airconditioning == 'yes' else 0,
                        parking,
                        1 if prefarea == 'yes' else 0,
                        0 if furnishingstatus == 'furnished' else 1 if furnishingstatus == 'semi-furnished' else 2]])

# Normalize the input data
input_data = (input_data - X.mean(axis=0)) / X.std(axis=0)

# Predict the house price
predicted_price = predict(input_data, W, b)

st.write(f"Predicted House Price: ${predicted_price[0][0]:,.2f}")
