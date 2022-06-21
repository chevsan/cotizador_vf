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
from sklearn.model_selection import train_test_split

######################################################################################################################

def cleaning_before_split_train(df,ventana,moneda,grupos):
    '''
    '''
    ### 0) Nulos ###
    df = df.dropna(subset=['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','Subseg_a', 'Seg_a'])
    
    ### 1) Duplicados ###
    # tratamiento de la feautre "runtime"
    df['runtime'] = pd.to_datetime(df.runtime.apply(lambda x: str(x)[:19]))
    df = df.sort_values(['runtime'])
    # teniendo el df ordenado, eliminamos los duplicados, quedandonos con el registro más reciente (la última ocurrencia)
    df.drop_duplicates(subset=['car_id'], keep='last', inplace=True)
    
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
    
    ### 4) Dropeamos 0kms y concesionarias
    df['dealer'] = np.where(df['dealer']==True,1,0)
    mask_not_0km = df.car_kms > 90
    mask_not_conces = df.dealer == 0
    df = df[(mask_not_0km) & (mask_not_conces)]
    
    ### 5) Dropeamos match_scores por debajo a 80% ###
    lst = ['score_marca_a','score_modelo_a','score_v1_c']
    for col in lst:
        df = df[df[col]>=80]
        
    ### 6) Ultimos (ventana) días ###
    df['runtime'] = df['runtime'].apply(pd.to_datetime)
    max_date = df.runtime.max()
    mask_window = (df.runtime <= max_date) & ((df.runtime >= max_date - timedelta(days=ventana)))
    df = df[mask_window]
    
    #### 7) Categorías que no nos interesa cotizar ###    
    # agrupacion
    df_grouped = df.groupby(['match_marca_a','match_modelo_a','match_v1_a'],as_index=False).size()
    df_grouped.columns = ['marca','modelo','version','cant']
    df_grouped.sort_values(by='cant',ascending=False,inplace=True)
    # deciles
    df_output = df_grouped.copy()
    data = df_grouped.copy()
    data.sort_values(by='cant',ascending=False,inplace=True)
    data['cuantiles' + str('_'+'cant')] =pd.qcut(data['cant'], 10, duplicates='drop')
    # tmp
    placeholder= 'marca'
    tmp = data.groupby('cuantiles' + str('_'+'cant')).agg({placeholder:'count'}).rename(columns={placeholder:'placeholder'})
    leni = len(tmp)
    tmp['cuantil' + str('_'+'cant')] = list(reversed(list(np.arange(1,leni+1,1))))
    tmp['bin' + str('_'+'cant')] = tmp.index
    tmp.reset_index(drop=True,inplace=True)
    tmp.drop('placeholder',1,inplace=True)
    # join
    data = data.merge(tmp, how='inner',left_on=['cuantiles' + str('_'+'cant')], right_on='bin' + str('_'+'cant'))
    # string marca-modelo-version en el df auxiliar
    data['marca_modelo_version'] = data['marca'] + str(' - ') +data['modelo'] + str(' - ') + data['version']
    # grupos df
    grupos_df = data.copy()
    # nos quedamos únicamente con los grupos que entran al modelo
    grupos_df_algor = grupos_df[grupos_df['cuantil_cant'].apply(lambda x: x in grupos)] # grupos tiene que ser una lista
    grupos_df_algor.rename(columns={'cuantil_cant':'grupos'},inplace=True)
    df_grupos = grupos_df_algor[['marca','modelo','version','marca_modelo_version', 'grupos']]
    marc_mod_vers_OK = list(df_grupos.marca_modelo_version.unique())
    # string marca-modelo-version en el df principal
    df['marca_modelo_version'] = df['match_marca_a'] + str(' - ') +df['match_modelo_a'] + str(' - ') + df['match_v1_a']
    mask = df.marca_modelo_version.apply(lambda x: x in marc_mod_vers_OK)
    df = df[mask]
    
    #### 8) Target: construccion final del target "price_meli_ok" ###
    
    if moneda == 'dolares':
        
        # Target: convertimos los precios en pesos a dolar y los que estan en dolares se mantienen como estan

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
        df['price_amount'] = np.where(df.price_symbol == '$', df.price_amount / df.blue_venta, df.price_amount) # pisamos price_amount con su version final la cual es el target
    
    elif moneda == 'pesos':
        
        # Target: Los precios en pesos quedan como estan y las filas que tienen precios en dolares se eliminan

        mask = df.price_symbol == 'U$S'
        df.drop(list(df[mask].index),axis=0,inplace=True)
        #df.rename(columns={'price_amount':'price_meli_ok'},inplace=True)  # finalmente le dejamos el nombre price_amount al target
    
    return df, marc_mod_vers_OK
    
    
    #### 9) Re-ordenamiento de las cols ### 
    id_features = ['runtime','car_id']
    model_features = ['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','marca_modelo_version','Subseg_a', 'Seg_a', 'price_amount']
    others = ['car_location_1', 'match_v1_c']
    df = df[id_features + model_features + others]
    

def cleaning_before_split_valid_score(df,ventana,moneda,grupos,marc_mod_vers_OK):
    '''
    '''
    ### 0) Nulos ###
    df = df.dropna(subset=['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','Subseg_a', 'Seg_a'])
    
    ### 1) Duplicados ###
    # tratamiento de la feautre "runtime"
    df['runtime'] = pd.to_datetime(df.runtime.apply(lambda x: str(x)[:19]))
    df = df.sort_values(['runtime'])
    # teniendo el df ordenado, eliminamos los duplicados, quedandonos con el registro más reciente (la última ocurrencia)
    df.drop_duplicates(subset=['car_id'], keep='last', inplace=True)
    
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
    
    ### 4) Dropeamos 0kms y concesionarias
    df['dealer'] = np.where(df['dealer']==True,1,0)
    mask_not_0km = df.car_kms > 90
    mask_not_conces = df.dealer == 0
    df = df[(mask_not_0km) & (mask_not_conces)]
    
    ### 5) Dropeamos match_scores por debajo a 80% ###
    lst = ['score_marca_a','score_modelo_a','score_v1_c']
    for col in lst:
        df = df[df[col]>=80]
        
    ### 6) Ultimos (ventana) días ###
    df['runtime'] = df['runtime'].apply(pd.to_datetime)
    max_date = df.runtime.max()
    mask_window = (df.runtime <= max_date) & ((df.runtime >= max_date - timedelta(days=ventana)))
    df = df[mask_window]
    
    #### 7) Categorías que no nos interesa cotizar ###
    df['marca_modelo_version'] = df['match_marca_a'] + str(' - ') +df['match_modelo_a'] + str(' - ') + df['match_v1_a']
    mask = df.marca_modelo_version.apply(lambda x: x in marc_mod_vers_OK)
    df = df[mask]
    
    #### 8) Target: construccion final del target "price_meli_ok" ###
    
    if moneda == 'dolares':
        
        # Target: convertimos los precios en pesos a dolar y los que estan en dolares se mantienen como estan

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
        df['price_amount'] = np.where(df.price_symbol == '$', df.price_amount / df.blue_venta, df.price_amount) # pisamos price_amount con su version final la cual es el target
    
    elif moneda == 'pesos':
        
        # Target: Los precios en pesos quedan como estan y las filas que tienen precios en dolares se eliminan

        mask = df.price_symbol == 'U$S'
        df.drop(list(df[mask].index),axis=0,inplace=True)
        #df.rename(columns={'price_amount':'price_meli_ok'},inplace=True)  # finalmente le dejamos el nombre price_amount al target
    
    return df
    
    
    #### 9) Re-ordenamiento de las cols ### 
    id_features = ['runtime','car_id']
    model_features = ['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','marca_modelo_version','Subseg_a', 'Seg_a', 'price_amount']
    others = ['car_location_1', 'match_v1_c']
    df = df[id_features + model_features + others]
    
    

    

######################################################################################################################

def cleaning_after_split_train(df_train, vd):
    '''fit transform train'''
    
    cols_to_clean = [vd, 'car_kms']
    
    ##############################################
    # 1) Outliers globales
    
    # creamos diccionarios para guardar la info de capeo de outliers
    thresh_outliers_glob = {}
    
    # dropmeamos outliers globales de price
    p_995 = df_train[cols_to_clean[0]].quantile(0.995)
    # guardamos la info de train para luego aplicarla en test
    thresh_outliers_glob['price_p995'] = p_995
    mask = df_train[cols_to_clean[0]] <= p_995
    df = df_train[mask]
    
    # Ahora lo mismo pero para kms
    p_995 = df_train[cols_to_clean[1]].quantile(0.995)
    # guardamos la info de train para luego aplicarla en test
    thresh_outliers_glob['kms_p995'] = p_995
    mask = df_train[cols_to_clean[1]] > p_995
    # dropeamos outliers globales de car_kms
    df_train = df_train[~mask]
    
    
    ##############################################
    # 2) Outliers por contexto
    # Dropeamos todos los outliers por contexo: "Tratamiento de outliers nº2"

    modelos = sorted(list(df_train.match_modelo_a.unique()))
    años = sorted(list(df_train.car_year.unique()))
    kms_thresh_outliers = {}
    price_thresh_outliers = {}
    old_shape = df_train.shape[0]
    for m in modelos:
        for a in años:

            modelo_año = m + '_' + str(a)


            # kms
            mask1 = df_train.match_modelo_a == m
            mask2 = df_train.car_year == a
            data = df_train[mask1 & mask2].copy()

            q1 = data.car_kms.quantile(0.25)
            q3 = data.car_kms.quantile(0.75)
            IQR = q3 - q1
            outl_thresh_superior = q3+3*IQR
            outl_thresh_inferior = q1-3*IQR
            kms_thresh_outliers[modelo_año] = (outl_thresh_inferior,outl_thresh_superior)
            filt_mask_sup = data.car_kms>kms_thresh_outliers[modelo_año][1]
            filt_mask_inf = data.car_kms<kms_thresh_outliers[modelo_año][0]
            data = data[~(filt_mask_sup | filt_mask_inf)]
            df_train = df_train.loc[~(mask1 & mask2),:]
            df_train = pd.concat([df_train,data],0)

            # price
            mask1 = df_train.match_modelo_a == m
            mask2 = df_train.car_year == a
            data = df_train[mask1 & mask2].copy()

            q1 = data.price_meli_ok.quantile(0.25)
            q3 = data.price_meli_ok.quantile(0.75)
            IQR = q3 - q1
            outl_thresh_superior = q3+3*IQR
            outl_thresh_inferior = q1-3*IQR
            price_thresh_outliers[modelo_año] = (outl_thresh_inferior,outl_thresh_superior)
            filt_mask_sup = data.price_meli_ok>price_thresh_outliers[modelo_año][1]
            filt_mask_inf = data.price_meli_ok<price_thresh_outliers[modelo_año][0]
            data = data[~(filt_mask_sup | filt_mask_inf)]
            df_train = df_train.loc[~(mask1 & mask2),:]
            df_train = pd.concat([df_train,data],0)   

    
    # Ultimo retoque: probamos tanto usando year como int y como float y la perfo del modelo dio apenas mejor con year en float
    df_train['car_year'] = df_train['car_year'].astype('float')
    
    
    return df_train, thresh_outliers_glob, price_thresh_outliers, kms_thresh_outliers



def cleaning_after_split_test_valid_score(df, vd, thresh_outliers_glob, price_thresh_outliers, kms_thresh_outliers):
    '''Transform test'''
    
    cols_to_clean = [vd, 'car_kms']
    
    # 1) Outliers globales
    # dropeamos outliers globales en price
    mask = df[cols_to_clean[0]] > thresh_outliers_glob['price_p995']
    df = df[~mask]
    # dropeamos outliers globales en kms
    mask = dfdf[cols_to_clean[1]] > thresh_outliers_glob['kms_p995']
    df = df[~mask]
    

    # 2) Outliers por contexto
    # upload the dictionary with the information (learned with the train set!) about the thresholds to cap outliers
    df['car_year'] = df['car_year'].astype('int')
    
    modelos = sorted(list(df.match_modelo_a.unique()))
    años = sorted(list(df.car_year.unique()))
    for m in modelos:
        for a in años:
            
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
    
    
def split(df,vd):
    model_features = ['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','match_v1_c', 'Subseg_a', 'Seg_a']
    id_feature = ['car_id']
    
    ###### Split #######
    X = df[model_features + id_feature]
    y = df[vd]

    # 200 bines para discretizar la variable continua
    bins = np.linspace(0, len(y), 200)
    # Save your Y values in a new ndarray,
    # broken down by the bins created above.
    y_binned = np.digitize(y, bins)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20,  stratify=y_binned,random_state=42)
    
    df_train = pd.concat([X_train,y_train],1)
    df_test = pd.concat([X_test,y_test],1)
    
    return df_train, df_test
