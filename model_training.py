import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import re
import math

# ============================
# 1. Load and clean dataset
# ============================

df = pd.read_csv("bengaluru_house_prices.csv")

df = df.dropna(subset=["size", "bath", "price", "total_sqft", "location"])

df["bhk"] = df["size"].apply(lambda x: int(str(x).split()[0]) if pd.notna(x) and str(x).split()[0].isdigit() else None)

def convert_sqft_to_num(x):
    try:
        if "-" in str(x):
            vals = [float(i) for i in str(x).split("-")]
            return (vals[0] + vals[1]) / 2
        elif str(x).replace('.', '', 1).isdigit():
            return float(x)
        else:
            nums = re.findall(r'\d+\.?\d*', str(x))
            return float(nums[0]) if nums else None
    except:
        return None

df["total_sqft"] = df["total_sqft"].apply(convert_sqft_to_num)
df = df.dropna(subset=["total_sqft"])

# ============================
# 2. Feature selection
# ============================
X = df[["total_sqft", "bath", "bhk", "location"]]
y = df["price"]

# ============================
# 3. Preprocessing
# ============================
numeric = ["total_sqft", "bath", "bhk"]
categorical = ["location"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

# ============================
# 4. Define models
# ============================
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

os.makedirs("models", exist_ok=True)
results = {}

# ============================
# 5. Train and evaluate models
# ============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

for name, model in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = math.sqrt(mse)  # RMSE calculation compatible with all versions
    
    results[name] = {
        "R2 Score": round(r2_score(y_test, y_pred), 4),
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "RMSE": round(rmse, 4)
    }
    
    joblib.dump(pipe, f"models/{name.replace(' ', '_').lower()}_pipeline.joblib")

# ============================
# 6. Display results
# ============================
print("\nModel Performance Comparison:\n")
print(pd.DataFrame(results).T)
