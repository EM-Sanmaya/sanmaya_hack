# Install dependencies (if not already installed)
!pip install xgboost pandas openpyxl scikit-learn

# Import required libraries
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset (Make sure you uploaded 'combinational_depth_10k.xlsx' in Colab)
file_path = "/content/combinational_depth_10k.xlsx"
df = pd.read_excel(file_path)

# Display dataset information
print("Dataset Shape:", df.shape)
print(df.head())

# Define target variable and features
target_column = "Combinational Depth Target"  # Your target column
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target

# Split dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=500,  # Number of trees
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"🔹 Mean Absolute Error (MAE): {mae:.4f}")
print(f"🔹 Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"🔹 R² Score: {r2:.4f}")

# Save trained model
model.save_model("xgboost_combinational_depth_model.json")
print("\n Model saved as 'xgboost_combinational_depth_model.json'")
