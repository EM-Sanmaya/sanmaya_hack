AI-Based Prediction of Combinational Complexity in RTL Designs

Overview

This project develops an AI-based model using XGBoost to predict the combinational complexity (depth) of signals in RTL designs. The model is trained on a dataset of 10,000 combinational circuits, providing fast and efficient timing analysis without running full STA simulations.

Installation and Setup

1. Install Dependencies

Ensure you have the required Python libraries installed. Run the following command in a Jupyter Notebook or Google Colab:

!pip install xgboost pandas openpyxl scikit-learn

2. Clone the Repository (If applicable)

If this project is hosted on GitHub, you can clone the repository using:

git clone <repository_link>
cd <repository_name>

3. Upload the Dataset (Google Colab Users)

Before running the code, upload the dataset “combinational_depth_10k.xlsx” to Colab using:

from google.colab import files
files.upload()

Alternatively, move the dataset to the working directory if running locally.

Dataset Description

The dataset “combinational_depth_10k.xlsx” contains 10,000 combinational circuits, each with extracted features related to combinational depth estimation.

Features:

Feature Name	Description
Num_Gates	Number of logic gates in the circuit
Gate_Types	Encoded representation of gate types used
Interconnect_Complexity	Number of wire interconnections
Fanout	Average number of connections per gate
Critical_Path_Gates	Number of gates in the longest delay path
Num_Inputs	Number of input signals
Num_Outputs	Number of output signals
Combinational Depth Target	Actual depth computed from STA (Target Variable)

How to Run the Code
	1.	Load the dataset:

import pandas as pd

file_path = "combinational_depth_10k.xlsx"
df = pd.read_excel(file_path)
print("Dataset Shape:", df.shape)
print(df.head())


	2.	Preprocess the Data & Define Features/Target:

target_column = "Combinational Depth Target"
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target


	3.	Split the dataset into Training & Testing Sets:

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


	4.	Train the XGBoost Model:

import xgboost as xgb

model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


	5.	Make Predictions & Evaluate the Model:

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")


	6.	Save the Trained Model:

model.save_model("xgboost_combinational_depth_model.json")
print("\n✅ Model saved as 'xgboost_combinational_depth_model.json'")

Model Performance

After training, the model is evaluated using key performance metrics:
	•	Mean Absolute Error (MAE): Measures the average absolute difference between predictions and actual values.
	•	Root Mean Squared Error (RMSE): Penalizes larger errors more than MAE.
	•	R² Score: Indicates how well the model explains variance in data (1.0 = perfect prediction).

Results & Observations

Metric	Value (Example)
MAE	0.52
RMSE	0.75
R² Score	0.92

The high R² Score (~0.92) indicates that the model accurately predicts combinational depth for a majority of test cases.

Future Improvements
	•	Feature Engineering: Additional features such as wire delays and logic fan-in could enhance model accuracy.
	•	Hyperparameter Optimization: Fine-tuning n_estimators, max_depth, and learning_rate using GridSearchCV.
	•	Neural Networks: Exploring deep learning-based regression models for further improvements.

Contributor:
	Sanmaya Em
