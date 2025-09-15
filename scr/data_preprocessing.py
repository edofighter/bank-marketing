# Preprocesamiento y Feature Engineering
# Módulo para preprocesamiento del dataset Bank Marketing.

#librerias e impotaciones }
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import pipeline as Imbpipeline

# Función para cargar datos
def get_features(df: pd.DataFrame, target: str = 'y', drop: list = None):
    if drop is None:
        drop =[] 
        features = df.drop(columns=[target] + drop)
        cat_features = features.select_dtypes(include=['object', 'category']).columns.tolist()
        num_features = features.select_dtypes(include=['int64', 'float64']).columns.tolist()
        return num_features,cat_features

#Devuelve un pipeline sklearn/imblearn con preprocesamiento
def build_preprocessing_pipeline(num_features, cat_features, use_smote:bool= True):
     
     #Transformadores
     num_transformer = StandardScaler()
     cat_transformer = OneHotEncoder(handle_unknown='ignore')
     
     #ColumnTransformer
     preprocessor = ColumnTransformer(
         transformers=[
             ('num', num_transformer, num_features),
             ('cat', cat_transformer, cat_features)
         ]
     )
     #Pipeline con SMOTE
     if use_smote:
         pipe = Imbpipeline(steps = [
             ('preprocessor', preprocessor),
         ])
     return pipe
 
 