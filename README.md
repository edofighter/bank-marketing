📊 Bank Marketing ML Project

Este repositorio contiene un proyecto de Machine Learning orientado a predecir la contratación de un depósito a plazo por parte de clientes bancarios, utilizando el dataset Bank Marketing (UCI)

🚀 Estructura del proyecto
bank-marketing-ml/
├── data/
│   ├── raw/         # Datos originales
│   └── processed/   # Datos procesados
├── notebooks/
│   ├── 01_explore_data.ipynb
│   ├── 02_eda_baseline.ipynb
│   ├── 03_models.ipynb
│   └── 04_results.ipynb
├── scripts/
│   └── get_data.py
├── models/          # Modelos entrenados
├── requirements.txt

data/raw → Conjunto de datos originales (CSV de OpenML, ID 1461).
data/processed → Datos limpios y preparados para el modelado.

⚠️ Nota importante:
La variable duration no debe incluirse en el entrenamiento final, ya que aporta información que genera fugas de datos (data leakage).

🔎 Flujo de trabajo
Descarga de datos → scripts/get_data.py
Exploración inicial → notebooks/01_explore_data.ipynb
EDA y modelo baseline → notebooks/02_eda_baseline.ipynb
Modelos avanzados → notebooks/03_models.ipynb
Comparación de resultados → notebooks/04_results.ipynb

📈 Resultados principales
Mejor modelo → XGBoost optimizado
Métrica AUC → ~0.85
Variables más relevantes → duración de la llamada, balance y edad

▶️ Ejecución
Clonar el repositorio y configurar el entorno:
git clone https://github.com/tu-usuario/bank-marketing-ml.git
cd bank-marketing-ml
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Ejecutar los notebooks en orden para reproducir el análisis y los modelos.
Los resultados y modelos entrenados se guardan automáticamente en las carpetas correspondientes.

🛠️ Tecnologías
Python → pandas, numpy, scikit-learn, matplotlib, seaborn, xgboost

