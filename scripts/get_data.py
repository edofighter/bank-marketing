#libreiras e imoportaciones
import argparse
import os
from pathlib import Path

from sklearn.datasets import fetch_openml

#Dataset OpenML bank-marketing id = 1461
DEFAULT_OUT = 'data/raw/bank_marketing.csv'
DEFAULT_ID = 1461

def main(out_path: str, data_id: int):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents = True, exist_ok = True)
    
    print(f'Descargando dataset OpenML id={data_id} ...')
    ds = fetch_openml(data_id = data_id, as_frame = True)
    X, y = ds.data, ds.target
    
    #DataFrame "crudo" sin modificar
    df = X.copy()
    df[ds.target.name if hasattr( ds, 'target') and ds.target is not None else 'y'] = y
    
    #Guardar
    df.to_csv(out_path, index = False, encoding = 'utf-8')
    print(f'Guardado: {out_path.resolve()}')
    print(f'Shape: {df.shape}')
    if 'y' in df.columns:
        print('Distribución de la variable objetivo (y):')
    print(df['y'].value_counts(dropna = False, normalize = True).round(4))
    
    #Aviso sobre 'duration' (fuga de info)
    if 'duration' in df.columns:
        print("Aviso: la columna 'duration' implica fuga de información. No la uses para entrenar el modelo final.")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Descarga el dataset Bank Marketing desde OpenML.')
    parser.add_argument('--out', default =  DEFAULT_OUT, help = 'Ruta de salida CSV (por defecto: data/raw/bank_marketing.csv)')
    parser.add_argument('--id', type = int, default = DEFAULT_ID, help = 'OpenML data_id (por defecto: 1461)' )
    args = parser.parse_args()
    main(args.out, args.id)