from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
import pandas as pd

import joblib

def save_model(model, model_name):
    joblib.dump(model, f"models/{model_name}.joblib")


def preprocess_data(data, target_variable, task_type):
    if target_variable not in data.columns:
        raise ValueError("La variable cible sélectionnée n'existe pas dans les données.")

    X = data.drop(target_variable, axis=1)
    y = data[target_variable]

   
    if task_type == "Classification" and y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    if X.empty:
        raise ValueError("Les données d'entrée sont vides après suppression de la variable cible.")

  
    X = pd.get_dummies(X)

    return X, y

def train_models(X, y, task_type):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {}
    performances = {}

    if task_type == "Classification":
        algorithms = {
            "Random Forest": RandomForestClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "SVM": SVC()
        }

        for name, model in algorithms.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            performances[name] = {
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
            models[name] = model

    else:
        algorithms = {
            "Random Forest": RandomForestRegressor(),
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "SVR": SVR()
        }

        for name, model in algorithms.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            performances[name] = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred),
            }
            models[name] = model

    return models, performances
