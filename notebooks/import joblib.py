import joblib

# Cargar modelos entrenados
rf = joblib.load("../models/random_forest.pkl")
xgb = joblib.load("../models/xgb_best.pkl")