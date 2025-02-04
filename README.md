# House Price Prediction

## Introduction
This project implements a house price prediction model using Linear Regression in Python. The model is trained on a dataset containing various house features and predicts the price based on user input.

## Features
- Uses `pandas` for data handling.
- Normalizes numerical features for better convergence.
- Encodes categorical variables using `LabelEncoder`.
- Implements gradient descent for training the linear regression model.
- Provides a Streamlit-based user interface for predictions.

## Dataset
The dataset used in this project is stored in `Housing.csv`, which contains house attributes such as:
- Area
- Number of bedrooms
- Number of bathrooms
- Stories
- Main road access
- Guest room availability
- Basement
- Hot water heating
- Air conditioning
- Parking spaces
- Preferred area
- Furnishing status
- Price (Target variable)

## Installation
Ensure you have the required dependencies installed:
```sh
pip install streamlit pandas numpy scikit-learn
```

## Running the Application
To run the Streamlit-based web application, execute:
```sh
streamlit run app.py
```
Replace `app.py` with the actual script name if different.

## Training the Model
The model is trained using gradient descent to optimize the weights and bias for the linear regression equation:
```
Y = W*X + b
```
To train the model manually, run:
```sh
python train.py
```

## Downloading the Model
To download the trained model, use the following code:
```python
import pickle

# Load the trained model
with open('house_price_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Save model to download
with open('downloaded_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("Model downloaded successfully!")
```

## Usage
1. Enter the house details in the Streamlit UI.
2. The model will normalize the inputs and predict the house price.
3. The predicted price will be displayed.

## Output Example
```
Predicted House Price: $200,000.00
```

## License
This project is open-source and available for modification and use under the MIT license.

