from pathlib import Path
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath('../src'))
from class_cotiz import Cotizador



BASE = Path().absolute().parent
RAW = Path(BASE/'raw')
NOTEBOOKS = Path(BASE/'notebooks')
DATA = Path(BASE/'datos')
REPORTS_PGM = Path(BASE/'reports_pgm')
MODELOS = Path(BASE/'modelos')
REPORTS = Path(BASE/'reports')
PERFORMANCE = Path(BASE/'performance')
DOCS = Path(BASE/'docs')


cotiz = Cotizador(name='kardur', ventana=15, moneda='dolares', grupos=[1])

cotiz = cotiz.load(input_route=MODELOS)



# load data oot
df_oot = pd.read_csv('{}/df_score_oot_mayor_2204.csv'.format(DATA))




# chequeo
print(df_oot.shape)
print(df_oot.date.min(), df_oot.date.max())
    


cotiz.predict(df=df_oot, type='validacion')
print(cotiz.df_eval_oot)