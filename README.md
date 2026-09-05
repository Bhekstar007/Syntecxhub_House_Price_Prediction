House Price Prediction

A baseline Linear Regression model that predicts house prices based on property features such as area, number of bedrooms/bathrooms, and amenities.

Project Overview

This project walks through a full, simple regression workflow:

Loading and exploring a housing dataset
Cleaning and encoding categorical features
Handling skewed data with log transformations
Training a Linear Regression model
Evaluating performance with RMSE and R^2
Interpreting model coefficients
Saving the trained model for reuse
Demonstrating example predictions
Dataset

The dataset (Housing.csv) contains 545 property records with the following features:

Column	Description
price	Sale price of the house (target variable)
area	Total area in square feet
bedrooms	Number of bedrooms
bathrooms	Number of bathrooms
stories	Number of stories
mainroad	Whether the house faces a main road (yes/no)
guestroom	Whether the house has a guest room (yes/no)
basement	Whether the house has a basement (yes/no)
hotwaterheating	Whether the house has hot water heating (yes/no)
airconditioning	Whether the house has air conditioning (yes/no)
parking	Number of parking spots
prefarea	Whether the house is in a preferred area (yes/no)
furnishingstatus	Furnishing status (furnished / semi-furnished / unfurnished)
Methodology
Data cleaning — categorical yes/no columns were mapped to 1/0, and furnishingstatus was one-hot encoded.
Skew correction — price and area were right-skewed, so both were log-transformed (price_log, area_log) to improve model performance.
Train/test split — 80/20 split with a fixed random seed for reproducibility.
Feature scaling — applied via StandardScaler to standardize inputs before training.
Model training — a LinearRegression model was trained on the log-transformed target.
Evaluation — predictions were converted back to real price units before calculating MAE, RMSE, and R^2.
Results
Metric	Value
MAE	~972,812
RMSE	~1,317,370
R^2	~0.657

The model explains roughly 66% of the variance in house prices, with an average prediction error of about 20% of the mean house price (~4,766,729).

Key Drivers of Price

Based on model coefficients, the strongest positive predictors of price were:

area_log (largest effect)
bathrooms
stories
airconditioning
prefarea

Unfurnished homes were associated with lower predicted prices relative to furnished ones.

Note: area and area_log were both included as features, introducing some multicollinearity since they represent the same underlying information. This is a known limitation of the current feature set.

Tech Stack
Python
pandas, numpy
matplotlib, seaborn (visualization)
scikit-learn (modeling, preprocessing, evaluation)
joblib (model persistence)
Project Structure
├── Housing.csv                     # Dataset
├── Housing_price_prediction.ipynb  # Full analysis and modeling notebook
├── house_price_model.pkl           # Saved trained model
├── scaler.pkl                      # Saved feature scaler
└── README.md
How to Run
Clone the repository and install dependencies:
bash
   pip install pandas numpy matplotlib seaborn scikit-learn joblib
Open Housing_price_prediction.ipynb in Jupyter or VS Code.
Run all cells in order from top to bottom.
Future Improvements
Address multicollinearity between area and area_log
Engineer or source additional features (e.g. location, property age)
Compare against non-linear models (e.g. Random Forest, Gradient Boosting) to capture relationships a linear model can't
