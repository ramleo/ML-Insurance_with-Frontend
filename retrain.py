#!/usr/bin/env python3
"""Retrain pipeline: drop id, engineer Policy Start Date, re-run GridSearchCV."""
import joblib, numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error

# ── 1. Load & feature-engineer ────────────────────────────────────────────────
df = pd.read_csv("data/Insurance.csv")

# Drop id — it's a row index, not a predictor
df = df.drop(columns=["id"])

# Parse Policy Start Date → numeric components, drop original
df["Policy Start Date"] = pd.to_datetime(df["Policy Start Date"], errors="coerce")
df["psd_year"]        = df["Policy Start Date"].dt.year
df["psd_month"]       = df["Policy Start Date"].dt.month
df["psd_day"]         = df["Policy Start Date"].dt.day
df["psd_day_of_week"] = df["Policy Start Date"].dt.dayofweek
df = df.drop(columns=["Policy Start Date"])

TARGET = "Premium Amount"
X = df.drop(columns=[TARGET])
y = df[TARGET].values

# ── 2. Split ──────────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Preprocessor ───────────────────────────────────────────────────────────
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(include="object").columns.tolist()

print(f"Numeric features  ({len(num_cols)}): {num_cols}")
print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  RobustScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# ── 4. Candidates & grids ────────────────────────────────────────────────────
candidates = {
    "Ridge": (
        Ridge(),
        {"model__alpha": [0.1, 1.0, 10.0]},
    ),
    "RandomForest": (
        RandomForestRegressor(random_state=42),
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 5, 10]},
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(random_state=42),
        {"model__n_estimators": [100, 200], "model__max_depth": [3, 5],
         "model__learning_rate": [0.05, 0.1]},
    ),
    "SVR": (
        SVR(),
        {"model__C": [1, 10], "model__kernel": ["rbf"]},
    ),
}

best_score, best_name, best_pipeline = -np.inf, None, None

for name, (model, param_grid) in candidates.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    gs = GridSearchCV(pipe, param_grid, cv=3, scoring="r2", n_jobs=-1)
    print(f"  Training {name}...", end=" ", flush=True)
    gs.fit(X_train, y_train)
    print(f"CV r2 = {gs.best_score_:.4f}  params={gs.best_params_}")
    if gs.best_score_ > best_score:
        best_score = gs.best_score_
        best_name  = name
        best_pipeline = gs.best_estimator_

# ── 5. Evaluate on test set ───────────────────────────────────────────────────
y_pred = best_pipeline.predict(X_test)
test_r2   = r2_score(y_test, y_pred)
test_rmse = mean_squared_error(y_test, y_pred) ** 0.5

print(f"\nBest model : {best_name}")
print(f"CV R²      : {best_score:.4f}")
print(f"Test R²    : {test_r2:.4f}")
print(f"Test RMSE  : {test_rmse:.2f}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
joblib.dump(best_pipeline, "models/final_pipeline.pkl")
print("\nSaved → models/final_pipeline.pkl")
print(f"Numeric cols : {num_cols}")
print(f"Cat cols     : {cat_cols}")
