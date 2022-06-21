#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from funcs_cotizador_vf import *
from funcs_gral import *
from IPython.display import display
from pathlib import Path
from time import strftime, localtime
from ipywidgets import widgets
import os
from os import listdir
from os.path import isfile, join
import pickle
import seaborn as sns  # Librería para visualización de datos
import matplotlib.pyplot as plt  # Librería para visualización de datos
import xlsxwriter  ## Libreria para dar formatos a archivos excel
import os,json
from sklearn import metrics
import itertools as it
from pandas.io import gbq
import seaborn as sns
import os,json
import matplotlib.pyplot as plt
from sklearn import metrics
import itertools as it
from datetime import date, datetime, timedelta
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")



class Cotizador:
    '''
    Objetivo: Generar un cotizador mediante un catboost.

    Descripcion: A partir de un dataframe de desarrollo, se eligen las variables a modelar
        y se desarrolla un modelo. Una vez que se obtiene
        el modelo deseado, se puede utilizar para predecir nuevos dataframes o leads. Si estos
        contienen la "vd_column", se tomarán como tablas de validación oot y se calcularán todos
        los atributos necesarios para el assesment. Si no la tiene, solo le agregará la "score_column" al final \
        (escenario de produccion).
    Aclaracion: A cualquier dataframe que le pasemos, ya sea desarrollo, validacion o simplemente scoreo, se le aplicara\
        el correspondiente tratamiento de limpieza y procesamiento y luego se haran las predicciones.
        
    Atributos por constructor (los que puede pasar el usuario):
        self.name = nombre de la clase
        self.ventana = ventana de dias del dataset
        self.moneda = indica si es modelo en pesos o en dolares
        self.grupos = grupos de presencialidad a incluir

    Atributos generados en el proceso
        self.df_eval_desarrollo = Metricas de performance de train y test en un mismo df
        self.marc_mod_vers = Contiene las marca-modelo-versiones que entraron dentro de los grupos de presencialidad seleccionados
        self.thresh_outliers_glob = thresholds the outliers globales para price y kms
        self.price_thresh_outliers = threshold de los outliers under context (marca-modelo-version) para price
        self.kms_thresh_outliers = threshold de los outliers under context (marca-modelo-version) para price
        self.catboost = modelo entrenado listo para hacer predicciones
        self.feature_importance = importancia de las variables segun el algortimo y el dataset de desarrollo que usamos para entrenar
        self.columns = nombre y orden de las columnas al momento del entrenamiento
        self.df_eval_oot = se genera unicamente si hacemos un .predict() con un dataset de validacion. Contiene la evaluacion correspondiente
        self.df_output_score = se genera unicamente si hacemos un .predict() con un dataset de scoreo. Contiene la cotizacion de los leads del dataset que le hayamos pasado
    '''
    def __init__(self, name, ventana, moneda, grupos):
        '''
        Args:
            name (string): Nombre para generar los objetos e identificarlos
            ventana
            moneda

        Output
            Sentencia de objeto creado correctamente
        '''

        self.name = name
        self.ventana = ventana
        self.moneda = moneda
        self.grupos = grupos
        
        print(f'Creando cotizador con el nombre {self.name}')
        print('\n')
        print(f'Ventana de {self.ventana} dias')
        print(f'Modelo en {self.moneda}')
        print(f'Grupos de presencialidad: {self.grupos}')

    def entrenamiento(self
                      ,df
                      ,vd_column
                      ,keep=None
                      ,drop=None
                      ,score_column='cotizacion'
                      ,random_state=0
                      ,print_results=True
                     ):
        '''
        Args:
            df (Pandas DataFrame): Tabla de datos a analizar
            vd_column (string): Nombre de la columna relacionada con una variable
                dependiente.
            keep (string o list, optional): Nombre de las columnas a mantener en
                el análisis. Defaults to None.
            drop (string o list, optional): Nombre de las columnas a excluir del
                análisis. Defaults to None.
            score_column (string, optional): Nombre que se le asignará a la columna
                de score. Importante solo si nuestra base contiene otra variable con
                el nombre dado por defecto. Default to 'score'.
            random_state (int, optional): Número de semilla pseudo-aleatoria. Defaults to 0.
            print_results (bool, optional): Imprime resultados.

        Output
            Hora de inicio, variables que ingresaron al análisis, variables que
            quedaron en el estuido y tablas de resumen y performance.
        '''
        self._vd_column = vd_column
        self._score_column = score_column
        self._random_state = random_state
        self._print_results = print_results

        print(f'START --- {strftime("%Y-%m-%d %H:%M:%S", localtime())}')
        print(f'  vd_column = {self._vd_column}')
        print(f'  keep = {keep}')
        print(f'  drop = {drop}')
        print(f'  score_column = {self._score_column}')
        print(f'  random_state = {self._random_state}')
        print(f'  print_results = {self._print_results}')
        print('\n')

        try:
            # Elimina las filas del drop
            if drop is not None:
                if isinstance(drop, str): drop = [drop]
                drop = drop.copy()
                if vd_column in drop:
                    drop.remove(vd_column)
                try:
                    df = df.drop(drop, axis=1)
                except:
                    raise (ValueError('Alguna de las columnas a dropear no se encontraban en el dataframe'\
                        'a fitear. Favor, verifique los nombres de las mismas.'))
            # Se queda solo con las filas en keep
            if keep is not None:
                if isinstance(keep, str): keep = [keep]
                keep = keep.copy()
                if vd_column not in keep:
                    keep.append(vd_column)
                try:
                    df = df[keep]
                except:
                    raise (ValueError('Alguna de las columnas a mantener no se encontraban en el dataframe'\
                        'a fitear. Favor, verifique los nombres de las mismas.'))

            if not df.index.is_unique:
                raise (ValueError('Existen índices duplicados para este Dataframe. Es necesario corregir' \
                                  'esto para que el proceso funcione.'))

        except:
            raise (ValueError(
                'No pudo generar el Cotizador. Verifique los parámetros ingresados.'))

        # Procesamiento
        #       
        # Cleaning before spliting (dataset desarrollo)
        df_clean, self.marc_mod_vers = cleaning_before_split_desarrollo(df=df, ventana=self.ventana, moneda=self.moneda, grupos=self.grupos)
        
        # Split
        # resolver el problema del nombre de la vd, tiene que ser el mismo de arranque (usaremos price_amount)
        df_train, df_test = split(df=df_clean, vd=self._vd_column)
        
        # Cleaning after spliting (train set)
        df_train_clean, self.thresh_outliers_glob, self.price_thresh_outliers, self.kms_thresh_outliers = cleaning_after_split_train(df_train= \
                                                                                                                     df_train
                                                                                                                      ,vd=self._vd_column)
        # Cleaning after spliting (test / validacion / data to score)
        df_test_clean = cleaning_after_split_test_valid(df_test, self._vd_column, self.thresh_outliers_glob, self.price_thresh_outliers, self.kms_thresh_outliers)
        
        # Train
        # aca se genera self.catboost
        train_pred, test_pred = run_model(self, entrenamiento=True, df_train=df_train_clean, df_test=df_test_clean)

        # Evaluacion del entrenamiento
        y_train = df_train_clean[self._vd_column]
        y_test = df_test_clean[self._vd_column]
        df_eval_train = evaluate(y_train, train_pred)
        df_eval_test = evaluate(y_test, test_pred)
        self.df_eval_desarrollo = pd.concat([df_eval_train,df_eval_test],axis=0, ignore_index=True)
        #self.df_eval_desarrollo['set'] = pd.Series(['train','test'])
        self.df_eval_desarrollo.index = ['train','test']

        # Nombre y orden de las columnas al momento del entrenamiento
        self.columns = list(df_train_clean.columns)

        if print_results == True:
            print('\n')
            print('El modelo fue entrenado con exito, a continuacion observaremos las metricas de performance:')
            print(self.df_eval_desarrollo)

    def predict(self, df, type='validacion'):
        '''
        Ejecuta el score a un nuevo dataframe. Si contiene la 'vd_column', considera que es una
        base de validación y calcula las evaluaciones.
        '''
        print(f'START --- {strftime("%Y-%m-%d %H:%M:%S", localtime())}')

        if type == 'validacion':
            # Processing
            df_clean1 = cleaning_before_split_valid(df=df,ventana=self.ventana,moneda=self.moneda,marc_mod_vers_OK=self.marc_mod_vers)
            df_clean2 = cleaning_after_split_test_valid(df=df_clean1, vd=self._vd_column
                                                    ,thresh_outliers_glob=self.thresh_outliers_glob, price_thresh_outliers=self.price_thresh_outliers
                                                    ,kms_thresh_outliers=self.kms_thresh_outliers)

            # Predict
            #breakpoint()
            val_oot_pred = run_model(self, entrenamiento=False, df_oot = df_clean2)  # al no ser entrenamiento, las opciones son que sea validacion oot o scoreo
            # Evaluate
            # y_true
            y_oot = df_clean2[self._vd_column]
            self.df_eval_oot = evaluate(y_oot, val_oot_pred)
            print(self.df_eval_oot)

        if type == 'scoreo':
            # Processing
            df_clean1 = cleaning_before_split_score(df,ventana=self.ventana,moneda=self.moneda,marc_mod_vers_OK=self.marc_mod_vers)
            df_clean2 = cleaning_after_split_score(df=df_clean1, vd=self.ventana
                                                    ,thresh_outliers_glob=self.thresh_outliers_glob, price_thresh_outliers=self.price_thresh_outliers
                                                    ,kms_thresh_outliers=self.kms_thresh_outliers)
            # Predict
            scoreo_pred = run_model(self, entrenamiento=False, df_score = df_clean2)  # al no ser entrenamiento, las opciones son que sea validacion oot o scoreo
            # Final output
            df_clean2[self._score_column] = scoreo_pred.copy()
            self.df_output_score = df_clean2.copy()
            #self.df_output_score = pd.concat([df_clean2, scoreo_pred],axis=1, ignore_index=True)

        print('\n')
        print(f'\nEND --- {strftime("%Y-%m-%d %H:%M:%S", localtime())}')
        if type == 'scoreo':
            return self.df_output_score.copy()

    def export(self, output_route='.'):
        '''
        Args:
            Output_route (string o pathlib object, optional): Path donde se quiera guardar el
                excel de los datos generados. Defaults to '.'.

        Output
            Info de donde lo guardó y bajo qué nombre
        '''

        print(f'START --- {strftime("%Y-%m-%d %H:%M:%S", localtime())}')
        exportar_a_excel(output_route, self)
        print(f'\nEND --- {strftime("%Y-%m-%d %H:%M:%S", localtime())}')

    def save(self, output_route='.'):
        '''
        Args
            Output_route (string o pathlib object, optional): Path donde se quiera
                guardar el pickle con el modelo. Defaults to '.'.

        Output
            Info de donde lo guardó y bajo qué nombre
        '''
        
        filename = 'Mod_' + self.name + '_' + strftime("%Y%m%d",
                                                       localtime()) + '.pkl'

        try:
            if isinstance(output_route,
                          str):  # Chechea si entramos con la ruta str o Path
                if output_route[-1] != '/': output_route = output_route + '/'
                with open(output_route + filename, 'wb') as file:
                    pickle.dump(self, file)
            else:
                with open(output_route / filename, 'wb') as file:
                    pickle.dump(self, file)

            print("Nombre del archivo:", filename)
            print("Guardado en:", Path(output_route).resolve())
        except:
            raise (ValueError(
                'Error con el Output_route. Verifique que sea correcto el directorio a guardar.'
            ))

    def load(self, input_route='.', filename=None):
        '''
        input_route (string o pathlib object, optional): Path desde donde se quiera levantar
            el modelo guardado. Defaults to '.'.
        filename (string, optional): Nombre del archivo que se quiera levantar.
            Por defecto (None), te busca un archivo que comienze con
            "Mod_nombrequeseledioalaclase_fechamáxima.pkl". Sin embargo, si el
            archivo tiene otro nombre se lo puede identificar con este parámetro.
            Defaults to None.

        Output:
            Prints de la hora de comienzo y filename a levantar.
        '''
        try:
            name = self.name
            if filename is None:
                lista_filenames = [
                    f for f in listdir(input_route)
                    if (isfile(join(input_route, f))) and (
                        f.startswith('Mod_' + name +
                                     '_')) and (f.endswith('.pkl'))
                ]
                lista_filenames.sort(reverse=True)
                filename = lista_filenames[0]
                print(f'El archivo que cargará es: {filename} .'\
                    '\nSi se quiere cargar otro archivo, puede hacerlo introduciendo '\
                        'el parámetro filename\n')
            if isinstance(input_route,
                          str):  # Chechea si entramos con la ruta str o Path
                if input_route[-1] != '/': input_route = input_route + '/'
                with open(input_route + filename, 'rb') as file:
                    return pickle.load(file)
            else:
                with open(input_route / filename, 'rb') as file:
                    return pickle.load(file)
        except:
            raise (ValueError(
                'Error con el input_route. Verifique que sea correcto el directorio del que se'\
                'quiere leer y/o especifique el nombre del archivo con el parámetro filename.'
            ))




########################################### FUNCIONES #############################################

def run_model(self, entrenamiento=False, df_train=None, df_test=None, df_oot = None, df_score = None):
    '''
    Función del proceso principal.
        - Si llega por "entrenamiento", entrena el modelo
        - Scorea (tanto en "entrenamiento" como en "predict")
        - Si el df tiene la vd_column, genera la evaluacion, si no la tiene es simplemente un scoreo
        - Devuelve el df scoreado si es transform
    '''


    if entrenamiento:
        print('Entrenando el modelo...')
        # Genera el modelo
        self.catboost = fit_catboost(
            df_train=df_train, df_test=df_test
            ,vd_column=self._vd_column
            ,random_state=self._random_state)
        # feature importance
        self.feature_importance = self.catboost.get_feature_importance(prettified=True)
        # Pred sobre train
        train_pred = self.catboost.predict(df_train.drop(labels=self._vd_column, axis=1))
        # Pred sobre test
        test_pred = self.catboost.predict(df_test.drop(labels=self._vd_column, axis=1))

    NoneType = type(None)
    # Validacion OOT
    if isinstance(df_oot,NoneType)==False:
        # Pred sobre validacion oot
        #breakpoint()
        val_oot_pred = self.catboost.predict(df_oot.drop(labels=self._vd_column, axis=1))

    # Scoreo (escenario de produccion)
    if isinstance(df_score,NoneType)==False:
        # Pred sobre validacion oot
        scoreo_pred = self.catboost.predict(df_score)  # como este df no viene con vd porque justamente es data nueva, no hace falta el .drop(vd)
        scoreo_pred = pd.Series(data=scoreo_pred, name=self._score_column)

    if (entrenamiento == False and isinstance(df_oot,NoneType)==False):
        return val_oot_pred  # estas serian las predicciones de validacion
    
    elif (entrenamiento == False and isinstance(df_score,NoneType)==False):
        return scoreo_pred  # estas serian las predicciones de scoreo

    elif entrenamiento == True:
        return train_pred, test_pred   # estas serian las predicciones de train/test (base de desarrollo) para poder hacer el assesment del entrenamiento


def fit_catboost(df_train, df_test
                  ,vd_column
                  ,random_state=0
                  ):
    '''
    Realiza el fit del modelo.
    '''
    model_features = ['car_year','car_kms','match_marca_a','match_modelo_a','match_v1_a','Subseg_a', 'Seg_a']

    # Volvemos a separar en X e y
    X_train = df_train[model_features]
    y_train = df_train[vd_column]

    X_test = df_test[model_features]
    y_test = df_test[vd_column]   

    # Genera el modelo
    model=CatBoostRegressor(loss_function='RMSE',logging_level="Silent",random_state=random_state)
    categorical_features_indices = np.where(X_train[model_features].dtypes != np.float)[0]
    model.fit(X_train, y_train, cat_features=categorical_features_indices)

    return model


def exportar_a_excel(output_route, self):
    '''
    Exporta a excel.
        - El resumen del modelo
        - Todas las tablas generadas para entrenar
        - Las últimas tablas generadas para validar (si las hubiera)
    '''
    # Escribimos las tablas, cada una a una tab distinta del excel.
    filename = 'Mod_' + self.name + '_' + strftime("%Y%m%d",
                                                   localtime()) + '.xlsx'

    try:
        if isinstance(output_route,
                      str):  # Chechea si entramos con la ruta str o Path
            if output_route[-1] != '/': output_route = output_route + '/'
            writer = pd.ExcelWriter(output_route + filename,
                                    engine='xlsxwriter')

        else:
            writer = pd.ExcelWriter(output_route / filename,
                                    engine='xlsxwriter')

        workbook = writer.book

        #Definimos formatos genéricos para después poder usarlos.
        pct_format = workbook.add_format({
            'num_format': '0.00%',
            'bg_color': 'white',
            'align': 'right'
        })
        int_format = workbook.add_format({
            'bg_color': 'white',
            'align': 'right'
        })
        str_format = workbook.add_format({
            'bg_color': 'white',
            'align': 'left'
        })
        header_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'bg_color': '#961B32',
            'font_color': 'white'
        })
        header_format_2 = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'bg_color': '#961B32',
            'font_color': 'white',
            'align': 'right'
        })
        header_format_3 = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'bg_color': '#961B32',
            'font_color': 'white',
            'font_size': 10,
            'align': 'right'
        })
        index_format = workbook.add_format({
            'bold': True,
            'text_wrap': True,
            'bg_color': 'white',
            'align': 'right'
        })
        bg_format = workbook.add_format({'font_size': 10, 'bg_color': 'white'})


        #self.df_eval_oot = se genera unicamente si hacemos un .predict() con un dataset de validacion. Contiene la evaluacion correspondiente
        #self.df_output_score = se genera unicamente si hacemos un .predict() con un dataset de scoreo. Contiene la cotizacion de los leads de
        
        ## self.df_eval_desarrollo ##
        try:
            self.df_eval_desarrollo.to_excel(writer
                                            ,sheet_name='Performance_dev'
                                            ,index=False)
            #worksheet_1 = writer.sheets['Performance_dev']
            #worksheet_1.set_column('A:G', 15, str_format)
        except:
            print('Error en la Performance_dev!')

        ## self.feature_importance ##
        try:
            self.marc_mod_vers.to_excel(writer,
                                      sheet_name='Marcas_modelos_vers',
                                      index=False)
            #worksheet_2 = writer.sheets['Summary_dev']
            #worksheet_2.set_column('A:A', 25, str_format)
            #worksheet_2.set_column('B:K', 15, int_format)
            #for col_num, value in enumerate(self.summary_dev.columns.values):
            #    worksheet_2.write(0, col_num, value, header_format)
        except:
            print('Error en el listado de marc_mod_vers')

        ## self.marc_mod_vers ##
        try:
            self.feature_importance.to_excel(writer,
                                      sheet_name='feature_importance',
                                      index=False)
        except:
            print('Error en feature_importance')

        ## self.df_eval_oot ##
        try:
            self.df_eval_oot.to_excel(writer
                                    ,sheet_name='Performance_oot'
                                    ,index=False)
        except:
            print('Error en la Performance_oot!')

        
        ## self.df_output_score ##
        try:
            self.df_output_score.to_excel(writer
                                    ,sheet_name='Output_score'
                                    ,index=False)
        except:
            print('Error en la Output_score!')



        writer.save()

        print("Nombre del archivo:", filename)
        print("Guardado en:", Path(output_route).resolve())
    except:
        raise (ValueError(
            'Error con el Output_route. Verifique que sea correcto el directorio a guardar.'
        ))


