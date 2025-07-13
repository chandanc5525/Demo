# Step 1: Import Neccessory Libraries
import pandas as pd
import joblib
import time 
import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()])

# Import Scikit Learn Libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Step 2: Data Loading:
def load_data(filepath, target):
    logging.info(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset.")
    X = df.drop(columns=target)
    y = df[target]
    return X, y

# Step 3: Data Preprocessing
def build_preprocessor(X):
    num_cols = X.select_dtypes(include="number").columns
    cat_cols = X.select_dtypes(exclude="number").columns

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])
    return preprocessor

# Step 4: Data Training Pipeline
def train_pipeline(X, y, task="classification", tune=False):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(X)

    model_cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
    model = model_cls(random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    if tune:
        param_grid = {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        }
        logging.info("Starting hyperparameter tuning...")
        search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1)
        search.fit(X_train, y_train)
        pipeline = search.best_estimator_
        logging.info(f"Best params: {search.best_params_}")
    else:
        pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    if task == "classification":
        score = accuracy_score(y_test, y_pred)
        logging.info(f"Test Accuracy: {score:.4f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logging.info(f"Test MSE: {mse:.4f}")
        logging.info(f"Test R²: {r2:.4f}")
    
    start = time.time()
    y_pred = pipeline.predict(X_test)
    end = time.time()
    logging.info(f"Prediction on test set took {(end - start):.4f} seconds")
    return pipeline

# Step 5: Saving Model
def save_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    logging.info(f"Pipeline saved to {path}")

# Step 6: Writing Pipeline Executions
if __name__ == "__main__":
    url = 'https://raw.githubusercontent.com/chandanc5525/EnE_HearDiesease_ClassificationModel/refs/heads/main/data/raw/heart-disease.csv'
    target_col = "target"

    # Load & train
    X, y = load_data(url, target=target_col)
    model = train_pipeline(X, y, task="classification", tune=True)

    # Save the trained pipeline
    save_pipeline(model, "modelpipeline.joblib")
