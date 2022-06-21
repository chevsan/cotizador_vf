import pandas as pd
import numpy as np
import seaborn as sns
import os,json
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools as it
import pandas as pd
import numpy as np
from pandas.io import gbq
import seaborn as sns
import os,json
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools as it
from datetime import date, datetime, timedelta

######################################################################################################################

def cleaning_before_split(df,ventana):
    '''Esta funcion se usa para las simulaciones.'''
    # parametros
    path_save = '../datos/'
    # dolar_blue = 197  # actualizado al 04/19 
    # dolar_oficial = 119.28
    
    old_shape = df.shape[0]
    ### 0) Nulos ###
    df = df.dropna(subset=['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','Subseg_a', 'Seg_a'])
    print(f'Hey! {old_shape - df.shape[0]} were removed due to null values')
    old_shape = df.shape[0]
    
    old_shape = df.shape[0]
    ### 1) Duplicados ###
    # tratamiento de la feautre "runtime"
    df['runtime'] = pd.to_datetime(df.runtime.apply(lambda x: str(x)[:19]))
    df = df.sort_values(['runtime'])
    # teniendo el df ordenado, eliminamos los duplicados, quedandonos con el registro más reciente (la última ocurrencia)
    df.drop_duplicates(subset=['car_id'], keep='last', inplace=True)
    
    print(f'Hey! {old_shape - df.shape[0]} were removed due to duplicate values')
    old_shape = df.shape[0]
    
    ### 2) 11111 & 99999 ###
    # llevamos a int para detectar más facilmente los 1111 o 9999
    df['price_amount'] = df.price_amount.astype('int')
    # mascaras para los 1111 de price
    m1 = df.price_amount == 1111
    m2 = df.price_amount == 11111
    m3 = df.price_amount == 111111
    m4 = df.price_amount == 1111111
    m5 = df.price_amount == 11111111
    m6 = df.price_amount == 111111111
    # dropeamos los 11111 de price
    df = df[~(m1 | m2 | m3 | m4 | m5 | m6)]
    # mascaras para los 9999 de price
    m1 = df.price_amount == 9999
    m2 = df.price_amount == 99999
    m3 = df.price_amount == 999999
    m4 = df.price_amount == 9999999
    m5 = df.price_amount == 99999999
    # dropeamos los 9999 de price
    df = df[~(m1 | m2 | m3 | m4 | m5)]
    # ahora lo mismo pero para kms
    # llevamos a int para detectar más facilmente los 1111 o 9999
    df['car_kms'] = df.car_kms.astype('int')
    # mascaras para los 1111 de kms
    m1 = df.car_kms == 1
    m2 = df.car_kms == 11
    m3 = df.car_kms == 111
    m4 = df.car_kms == 1111
    m5 = df.car_kms == 11111
    m6 = df.car_kms == 111111
    m7 = df.car_kms == 1111111
    m8 = df.car_kms == 11111111
    # dropeamos los 1111 de kms
    df = df[~(m1 | m2 | m3 | m4 | m5 | m6 | m7 | m8)]
    # mascaras para los 9999 de kms
    m1 = df.car_kms == 999
    m2 = df.car_kms == 9999
    m3 = df.car_kms == 99999
    m4 = df.car_kms == 999999
    m5 = df.car_kms == 9999999
    # dropeamos los 999 de kms
    df = df[~(m1 | m2 | m3 | m4 | m5)]
    
    print(f'Hey! {old_shape - df.shape[0]} were removed due those 11111 or 9999 strange values')
    old_shape = df.shape[0]
    
    ### 3) Construccion final del target ###
    # Construcción del precio final
    # blue= dolar_blue
    # oficial= dolar_oficial
    # col1 = 'price_symbol'
    # col2 = 'car_kms'
    # conditions = [df[col1]!='U$S', (df[col1]=='U$S') & (df[col2]==0), (df[col1]=='U$S') & (df[col2]!=0)]
    # choices = [df.price_amount, df['price_amount']*oficial, df['price_amount']*blue]
    # df['price_meli_ok'] = np.select(conditions, choices, default=np.nan)
    #----
    #mask = df.price_symbol == 'U$S'
    #df.drop(list(df[mask].index),axis=0,inplace=True)
    #df.rename(columns={'price_amount':'price_meli_ok'},inplace=True)
    #print(f'Hey! {old_shape - df.shape[0]} were removed due prices in dollars')
    
    old_shape = df.shape[0]
    ### 4) Dropeamos 0kms y concesionarias
    df['dealer'] = np.where(df['dealer']==True,1,0)
    mask_not_0km = df.car_kms > 90
    mask_not_conces = df.dealer == 0
    df = df[(mask_not_0km) & (mask_not_conces)]
    
    print(f'Hey! {old_shape - df.shape[0]} were removed due to 0km or concesioarias')
    old_shape = df.shape[0]
    
    ### 5) Dropeamos match_scores por debajo a 80% ###
    lst = ['score_marca_a','score_modelo_a','score_v1_c']
    for col in lst:
        df = df[df[col]>=80]
        
    print(f'Hey! {old_shape - df.shape[0]} were removed due to match scores under 80%')
    old_shape = df.shape[0]
        
    ### 6) Ultimos 15(ventana) días ###
    df['runtime'] = df['runtime'].apply(pd.to_datetime)
    max_date = df.runtime.max()
    mask_window = (df.runtime <= max_date) & ((df.runtime >= max_date - timedelta(days=ventana)))
    df = df[mask_window]
    
    print(f'Hey! {old_shape - df.shape[0]} were removed due to last 15d filter')
    old_shape = df.shape[0]
    
    #### 7) Categorías que no nos interesa cotizar ###    
    marc_mod_vers_OK = pd.read_csv('{}marc_mod_vers_OK.csv'.format(path_save), index_col=[0])
    marc_mod_vers_OK = list(marc_mod_vers_OK.iloc[:,0])
    mask = df.marca_modelo_version.apply(lambda x: x in marc_mod_vers_OK)
    df = df[mask]
    
    print(f'Hey! {old_shape - df.shape[0]} were removed due to categories in which we are not interested in score')
    
    
    #### 8) Target: convertimos los precios en pesos a dolar ###

    # Columna que nos indique dia de semana para ver cuando es finde
    df['date'] = pd.to_datetime(df['date'])
    df['dia_sem'] = df.date.dt.day_of_week
    dic = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
    df['dia_sem'] = df.dia_sem.map(dic)
    df.dia_sem.unique()
    # Ahora crearemos una que le ponga la fecha del viernes anterior a todos los dias que sean findesemana
    df['date_ok'] = np.where(df['dia_sem'].apply(lambda x: x in ('Saturday','Sunday'))
                             ,np.where(df['dia_sem']=='Saturday'
                                       ,df['date'] - timedelta(1)
                                       ,df['date'] - timedelta(2))
                             ,df['date'])
    # Importar tabla de Big Query
    filter_date = '2022-01-01'
    query = '''
                select *
                from `data-team-k.pricing_data.dolar_values` 
                where date >= '{}'
                and source = 'Blue'
                order by date

                '''.format(filter_date)
    df_dolar_values = gbq.read_gbq(query, project_id="data-team-k")
    df_dolar_values.columns = [col.lower() for col in df_dolar_values]
    # Ahora si procedemos a hacer el join con el df_dolar_values
    df = df.merge(df_dolar_values, how='inner', left_on='date_ok', right_on='date')
    df.drop(['date_y','source','compra'],1,inplace=True)
    df.rename(columns={'venta':'blue_venta'},inplace=True)
    # Finalmente, creamos la columna de precio final en dolares
    ## Los precios en dolares se mantienen como estan y los precios en pesos se pasan a dolar mediante el blue
    df['price_meli_ok'] = np.where(df.price_symbol == '$', df.price_amount / df.blue_venta, df.price_amount)
    
    return df
    
    
    #### 9) Ultimos retoques ###  
    # cuando hice el tratamiento de 1111 y 99999 había pasado la feature de kms a int. Volvemos a pasar a float por el catboost
    #df['car_kms'] = df['car_kms'].astype('float') # lo hacemos en la prox func de procesamiento
    df['car_year'] = df['car_year'].astype('int')
    
    id_features = ['runtime','car_id']
    model_features = ['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','marca_modelo_version','Subseg_a', 'Seg_a', 'price_meli_ok']
    others = ['car_location_1', 'match_v1_c']
    df = df[id_features + model_features + others]
    
    

    

######################################################################################################################


def cleaning_after_split(df, path_data):
    '''This function is to perform the data cleaning on a test or validation set'''
    # parametros
    path_save = '../datos/'
    
    # 1) Outliers globales
    old_shape = df.shape[0]
    f = open(os.path.join(path_save,'thresh_outliers_1.json'))
    thresh_outliers_1 = json.load(f)
    # dropeamos outliers globales en price
    mask = df.price_meli_ok > thresh_outliers_1['price_p995']
    df = df[~mask]
    # dropeamos outliers globales en kms
    mask = df.car_kms > thresh_outliers_1['kms_p995']
    df = df[~mask]
    
    print(f'Hey! {old_shape - df.shape[0]} were removed due to outliers globales')
    
    
    
    
    # 2) Outliers por contexto
    # upload the dictionary with the information (learned with the train set!) about the thresholds to cap outliers
    f = open(os.path.join(path_data,'price_thresh_outliers.json'))
    price_thresh_outliers = json.load(f)
    f = open(os.path.join(path_data,'kms_thresh_outliers.json'))
    kms_thresh_outliers = json.load(f)
    df['car_year'] = df['car_year'].astype('int')
    
    modelos = sorted(list(df.match_modelo_a.unique()))
    años = sorted(list(df.car_year.unique()))
    old_shape = df.shape[0]
    for m in modelos:
        for a in años:
            # print(f'{m} of {a}') --> solo para chequear que el loop este iterando correctamente (esta OK :)
            
            modelo_año = m + '_' + str(a)
            
            if modelo_año not in list(price_thresh_outliers.keys()):
                continue
            elif (str(price_thresh_outliers[modelo_año][0]) == 'nan'):
                continue
            else:
                # kms
                mask1 = df.match_modelo_a == m
                mask2 = df.car_year == a
                data = df[mask1 & mask2].copy()

                filt_mask_sup = data.car_kms>kms_thresh_outliers[modelo_año][1]
                filt_mask_inf = data.car_kms<kms_thresh_outliers[modelo_año][0]
                data = data[~(filt_mask_sup | filt_mask_inf)]
                df = df.loc[~(mask1 & mask2),:]
                df = pd.concat([df,data],0)

                # price
                mask1 = df.match_modelo_a == m
                mask2 = df.car_year == a
                data = df[mask1 & mask2].copy()

                filt_mask_sup = data.price_meli_ok>price_thresh_outliers[modelo_año][1]
                filt_mask_inf = data.price_meli_ok<price_thresh_outliers[modelo_año][0]
                data = data[~(filt_mask_sup | filt_mask_inf)]
                df = df.loc[~(mask1 & mask2),:]
                df = pd.concat([df,data],0)
    
    # probamos tanto usando year como int y como float y la perfo del modelo dio apenas mejor con year en float
    df['car_year'] = df['car_year'].astype('float')
    
    print(f'Hey! {old_shape - df.shape[0]} were removed from df due to outliers under context')  
    
    return df

######################################################################################################################

def print_evaluate(true, predicted):
    
    def mape(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        mape = np.mean(np.abs((actual-pred)/actual)) * 100
        return mape
    def medape(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        medape = np.median(np.abs((actual-pred)/actual)) * 100
        return medape
    
    mae = metrics.mean_absolute_error(true, predicted)
    mape = mape(true, predicted)
    medape = medape(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    print('MAE:', mae)
    print('MAPE:', mape)
    print('MEDAPE:', medape)
    print('MSE:', mse)
    print('RMSE:', rmse)
    print('R2 Square', r2_square)
    print('__________________________________')
    
def evaluate(true, predicted):
    mae = metrics.mean_absolute_error(true, predicted)
    mse = metrics.mean_squared_error(true, predicted)
    rmse = np.sqrt(metrics.mean_squared_error(true, predicted))
    r2_square = metrics.r2_score(true, predicted)
    return mae, mse, rmse, r2_square

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in it.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label discretized')
    plt.xlabel('Predicted label discretized')
    plt.show()
