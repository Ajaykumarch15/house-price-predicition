import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
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

# Convert X to a numpy array for further processing
X = X.values


# Initialize weights and bias for the Linear Regression model
def initialize_parameters(n):
    W = np.zeros((n, 1))
    b = 0
    return W, b


# Model Prediction (Hypothesis Function)
def predict(X, W, b):
    return np.dot(X, W) + b


# Compute cost (Mean Squared Error)
def compute_cost(Y_pred, Y):
    m = len(Y)
    cost = (1 / (2 * m)) * np.sum((Y_pred - Y) ** 2)
    return cost


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
        if i % 100 == 0:
            print(f"Iteration {i}, Cost: {compute_cost(Y_pred, Y)}")
    return W, b


# Train the linear regression model
W, b = train(X, Y, learning_rate=0.01, iterations=1000)

# Make predictions
Y_pred = predict(X, W, b)

# Display the first few predictions along with actual values
print("\nPredictions vs Actual values:")
for pred, actual in zip(Y_pred[:5], Y[:5]):
    print(f"Predicted: {pred[0]:.2f}, Actual: {actual[0]}")
