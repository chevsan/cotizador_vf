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
import warnings
warnings.filterwarnings("ignore")



class Cotizador:
    '''
    Objetivo: Generar un cotizador mediante un catboost.

    Descripcion: A partir de un dataframe de desarrollo, se eligen las variables a modelar
        y se desarrolla un modelo. Una vez que se obtiene
        el modelo deseado, se puede utilizar para predecir nuevos dataframes o leads. Si estos
        contienen la "vd_column", se tomarán como tablas de validación oot y se calculará todos
        los atributos necesarios para el assesment. Si no la tiene, solo le agregará la "score_column" al final \
        (escenario de produccion).
    Aclaracion: A cualquier dataframe que le pasemos, ya sea desarrollo, validacion o simplemente scoreo, se le aplicara\
        el correspondiente tratamiento de limpieza y procesamiento.
        
    Atributos por constructor:
        self.name = nombre de la clase
        self.ventana = ventana de dias del dataset
        self.moneda = indica si es modelo en pesos o en dolares
        self.grupos = grupos de presencialidad a incluir

    Atributos generados
        self.model = modelo treebased con gradient boosting (catboost).
        self.summary_dev = Tabla de resumen de la base de desarrollo.
        self.performance_dev = Metricas de performance de train.
        self.performance_val = Metricas de performance de test.
        self.marc_mod_vers = Contiene las marca-modelo-versiones que entraron dentro de los grupos de presencialidad seleccionados
        self.thresh_outliers_glob = thresholds the outliers globales para price y kms
        self.price_thresh_outliers = threshold de los outliers under context (marca-modelo-version) para price
        self.kms_thresh_outliers = threshold de los outliers under context (marca-modelo-version) para price
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
                      ,score_column='Score'
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
                'No pudo generar el Modeling. Verifique los parámetros ingresados.'))
        
        
        # Cleaning before spliting (dataset desarrollo)
        df_clean, self.marc_mod_vers = cleaning_before_split_train(df=df,ventana=self.ventana,moneda=self.moneda,grupos=self.grupos)
        
        # Split
        # resolver el problema del nombre de la vd, tiene que ser el mismo de arranque
        df_train, df_test = split(df=df_clean, vd=self._vd_column)
        
        # Cleaning after spliting (train)
        df_train_clean, self.thresh_outliers_glob, self.price_thresh_outliers, self.kms_thresh_outliers = cleaning_after_split_train(df_train= \
                                                                                                                     df_train
                                                                                                                      ,vd=self._vd_column)
        # Cleaning after spliting (test)
        df_test_clean = cleaning_after_split_test_valid_score(df_test, self._vd_column, self.thresh_outliers_glob, self.price_thresh_outliers, self.kms_thresh_outliers)
        
        
        # Train
        # llamar a una func definida afuera de la clase
        run_model(df)
        
        
        # Assement
        evaluate()

        
        
    def predict(self, df):
        '''
        Ejecuta el score a un nuevo dataframe. Si contiene la 'vd_column', considera que es una
        base de validación y calcula las tablas de performance.
        '''
        print(f'START --- {strftime("%Y-%m-%d %H:%M:%S", localtime())}')
        try:
            df = run_model(self, df, fit=False)

        except:
            raise (ValueError('Alguna de las columnas del modelo entrenado no se encuentran en este nuevo '\
                         'dataframe. Favor verifique que todo esté correctamente generado.'))
        print('\n')
        print(f'\nEND --- {strftime("%Y-%m-%d %H:%M:%S", localtime())}')
        return df.copy()

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
        # Escribimos las tablas, cada una a una tab distinta del excel.
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



## FUNCIONES ##

def run_model(self, df, entrenamiento=True):
    '''
    Función del proceso principal.
        - Si llega por "entrenamiento", entrena el modelo
        - Scorea (tanto en "entrenamiento" como en "predict")
        - Si el df tiene la vd_column, genera la evaluacion, si no la tiene es simplemente un scoreo
        - Devuelve el df scoreado si es transform
    '''

    if entrenamiento:
        print('Calculando modelo...')
        # Genera el modelo
        self.catboost = fit_catboost(
            df=df,
            vd_column=self._vd_column,
            random_state=self._random_state)

    print('Scoreando modelo...')
    df_aux = check_na(df[self.binomial_model.params.index])

    # Scorea
    df = pd.concat([
        df,
        pd.DataFrame(
            ((self.binomial_model.predict(df_aux)) * 1000).astype(int),
            columns=[self._score_column],
            index=df_aux.index)
    ],
                   axis=1,
                   join='inner')

    if self._vd_column in df.columns:
        # Genera tabla de performance
        tabla_performance_aux = performance_tabla(p_dt=df, 
                                                  p_score=self._score_column, 
                                                  p_q=self._quantile_qty, 
                                                  p_gb=self._vd_column, 
                                                  tie=False)
           
        
        grafico_performance_aux = performance_grafico(p_dt=df,
                                                      gain_table=tabla_performance_aux,
                                                      p_score=self._score_column,
                                                      p_gb=self._vd_column,
                                                      p_q=self._quantile_qty,
                                                      benchmark=None,
                                                      benchmark_table=None)

        # Genera el resumen
        summary, correlation = resumen_modelo(df,
                                              self.binomial_model,
                                              self._vd_column,
                                              signific=self._signific,
                                              level_vif=self._level_vif,
                                              intercept=self._intercept_column)

        # Calcula el resumen del summary
        wrongp = list(summary[summary['wrongp'] == 1]["Variables"])
        wrongvif = list(summary[summary['wrongvif'] == 1]["Variables"])
        wrongsign = list(summary[summary['wrongsign'] == 1]["Variables"])

        print('Max Ks:', str(round(max(tabla_performance_aux['ks']),2)))
        print(f'Quedaron {len(wrongp)} variables con wrongp:')
        print(wrongp)
        print(f'Quedaron {len(wrongvif)} variables con wrongvif:')
        print(wrongvif)
        print(f'Quedaron {len(wrongsign)} variables con wrongsign:')
        print(wrongsign)

        # Imprime lo importante
        if self._print_results:
            if fit == True: aux = 'dev'
            if fit == False: aux = 'val'
            print('\n\n' + color.BOLD + 'summary_' + aux + color.END)
            display(summary)

            print('\n\n' + color.BOLD + 'self.tabla_performance_' + aux +
                  color.END)
            display_formatted_table(tabla_performance_aux)

            print('\n\n' + color.BOLD + 'self.grafico_performance_' + aux +
                  color.END)
            plt.show(grafico_performance_aux)

            if fit == True:
                print('\n' + color.BOLD + 'self.binomial_model.summary()' +
                      color.END + '\n')
                print(self.binomial_model.summary())

            print('\n\n' + color.BOLD + 'self.correlation_' + aux + color.END)
            display(correlation.style.background_gradient(cmap='coolwarm'))

        if fit == True:
            self.summary_dev = summary
            self.tabla_performance_dev = tabla_performance_aux
            self.grafico_performance_dev = grafico_performance_aux
            self.correlation_dev = correlation

        if fit == False:
            self.summary_val = summary
            self.tabla_performance_val = tabla_performance_aux
            self.grafico_performance_val = grafico_performance_aux
            self.correlation_val = correlation

    if fit == False:
        return df.copy()


def fit_catboost(df,
                  vd_column,
                  random_state=0):
    '''
    Realiza el fit del modelo.
    '''
    X = df.drop(vd_column, axis=1)
    y = df[vd_column]

    # Genera el modelo
    catboost_model = # copiar lo de la nb de train de catboost
    
    
    
    sm.GLM(y,
                            X,
                            family=sm.families.Binomial(),
                            freq_weights=pesos,
                            random_state=random_state).fit()

    return catboost_model






def resumen_modelo(df,
                   regre,
                   vd,
                   signific=0.05,
                   level_vif=2.5,
                   intercept='intercept'):
    '''
    Genera la lista del modelo.
    '''
    lista = pd.Series(dtype='float64')

    lista_variables = list(regre.params.index)
    lista_variables.remove(intercept)

    for n in df[lista_variables]:
        lista[n] = df[vd].corr(df[n], method='pearson')

    corre = pd.DataFrame(lista).rename(columns={
        0: 'corr'
    }).rename_axis('Variables').reset_index()
    para = pd.DataFrame(regre.params).rename(columns={
        0: 'Coef.'
    }).rename_axis('Variables').reset_index().query(
        "Variables!='const'").reset_index(drop=True)
    pvalue = pd.DataFrame(regre.pvalues).rename(columns={
        0: 'p-value'
    }).rename_axis('Variables').reset_index().query(
        "Variables!='const'").reset_index(drop=True)
    vif_data = pd.DataFrame()
    vif_data["Variables"] = lista_variables
    vif_data["vif"] = [
        variance_inflation_factor(df[lista_variables].values, i)
        for i in range(len(df[lista_variables].columns))
    ]
    minimo = pd.DataFrame(df[lista_variables].min()).rename(columns={
        0: 'min'
    }).rename_axis('Variables').reset_index()
    maximo = pd.DataFrame(df[lista_variables].max()).rename(columns={
        0: 'max'
    }).rename_axis('Variables').reset_index()

    lista2 = pd.Series(dtype='float64')
    for n in df[lista_variables]:
        lista2[n] = df[vd].corr(df[n], method='pearson')
    corre_signo = reg_signo(lista2).rename(
        columns={'signo_coef': 'signo_corr'})

    tbl_final = pd.merge(
        pd.merge(
            pd.merge(
                pd.merge(
                    pd.merge(
                        para,
                        pd.merge(pd.merge(corre, reg_signo(regre.params)),
                                 corre_signo)), pvalue), vif_data), minimo),
        maximo)
    tbl_final["wrongsign"] = np.where(
        tbl_final["signo_coef"] == tbl_final["signo_corr"], 1, 0)
    tbl_final["wrongp"] = np.where(tbl_final["p-value"] < signific, 0, 1)
    tbl_final["wrongvif"] = np.where(tbl_final["vif"] < level_vif, 0, 1)
    tbl_final["Importancia"] = tbl_final["Coef."] * (tbl_final["max"] -
                                                     tbl_final["min"])
    datafin = tbl_final[[
        "Variables", "Coef.", "corr", "wrongsign", "p-value", "wrongp", "vif",
        "wrongvif", "min", "max", "Importancia"
    ]]

    datacorre = df[lista_variables +
                   [vd]].corr(method='pearson').round(decimals=2)

    return datafin, datacorre











###############
## EXCEL ##

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

        try:
            pd.concat([
                pd.DataFrame(self.binomial_model.summary().tables[0]),
                pd.DataFrame(data=['']),
                pd.DataFrame(self.binomial_model.summary().tables[1])
            ]).to_excel(writer, sheet_name='Model', index=False, header=False)
            worksheet_1 = writer.sheets['Model']
            worksheet_1.set_column('A:G', 15, str_format)
        except:
            print('Error en el Model')

        try:
            self.summary_dev.to_excel(writer,
                                      sheet_name='Summary_dev',
                                      index=False)
            worksheet_2 = writer.sheets['Summary_dev']
            worksheet_2.set_column('A:A', 25, str_format)
            worksheet_2.set_column('B:K', 15, int_format)
            for col_num, value in enumerate(self.summary_dev.columns.values):
                worksheet_2.write(0, col_num, value, header_format)
        except:
            print('Error en el Summary')

        try:
            tabla_performance_dev = self.tabla_performance_dev
            filename_split = filename.split('.')[0] + '_dev'
            if isinstance(output_route,
                      str):  # Chechea si entramos con la ruta str o Path
                if output_route[-1] != '/': output_route = output_route + '/'
                self.grafico_performance_dev.savefig(output_route + filename_split +'.png', bbox_inches='tight') 
            else:
                self.grafico_performance_dev.savefig(output_route / (filename_split +'.png'), bbox_inches='tight')
            
            format_excel(gain_table=tabla_performance_dev,
                     sheet_name='Performance_dev',
                     writer=writer,
                     output_route=output_route,
                     filename=filename_split,
                     p_q=self._quantile_qty,
                     p_gb=self._score_column,
                     benchmark_flag=False)
#             tabla_performance_dev.to_excel(writer,
#                                            sheet_name='Performance_dev',
#                                            index=False)
#             worksheet_3 = writer.sheets['Performance_dev']
#             worksheet_3.set_column('A:T', 12, int_format)
#             for col_num, value in enumerate(
#                     tabla_performance_dev.columns.values):
#                 worksheet_3.write(0, col_num, value, header_format)

#             ubicacion_graph = 'B' + str(len(tabla_performance_dev) + 5)
#             self.grafico_performance_dev.savefig(
#                 'tmp_figure.png', bbox_inches='tight')  ## Guarda imagen
#             worksheet_3.insert_image(ubicacion_graph, 'tmp_figure.png', {
#                 'x_scale': 0.70,
#                 'y_scale': 0.70
#             })
        except:
            print('Error en Performance_dev')

        try:
            self.correlation_dev.to_excel(writer,
                                          sheet_name='Correlaciones_dev')
            worksheet_4 = writer.sheets['Correlaciones_dev']
        except:
            print('Error en Correlaciones dev')

        try:
            self.summary_val.to_excel(writer,
                                      sheet_name='Summary_val',
                                      index=False)
            worksheet_5 = writer.sheets['Summary_val']
            worksheet_5.set_column('A:A', 25, str_format)
            worksheet_5.set_column('B:K', 15, int_format)
            for col_num, value in enumerate(self.summary_val.columns.values):
                worksheet_5.write(0, col_num, value, header_format)
        except:
            pass

        try:
            tabla_performance_val = self.tabla_performance_val
            filename_split = filename.split('.')[0] + '_val'
            if isinstance(output_route,
                      str):  # Chechea si entramos con la ruta str o Path
                if output_route[-1] != '/': output_route = output_route + '/'
                self.grafico_performance_val.savefig(output_route + filename_split +'.png', bbox_inches='tight') 
            else:
                self.grafico_performance_val.savefig(output_route / (filename_split +'.png'), bbox_inches='tight')
            
            format_excel(gain_table=tabla_performance_val,
                     sheet_name='Performance_val',
                     writer=writer,
                     output_route=output_route,
                     filename=filename_split,
                     p_q=self._quantile_qty,
                     p_gb=self._score_column,
                     benchmark_flag=False)
#             tabla_performance_val.to_excel(writer,
#                                            sheet_name='Performance_val',
#                                            index=False)
#             worksheet_6 = writer.sheets['Performance_val']
#             worksheet_6.set_column('A:T', 12, int_format)
#             for col_num, value in enumerate(
#                     tabla_performance_val.columns.values):
#                 worksheet_6.write(0, col_num, value, header_format)

#             ubicacion_graph = 'B' + str(len(tabla_performance_val) + 5)
#             self.grafico_performance_val.savefig(
#                 'tmp_figure.png', bbox_inches='tight')  ## Guarda imagen
#             worksheet_6.insert_image(ubicacion_graph, 'tmp_figure.png', {
#                 'x_scale': 0.70,
#                 'y_scale': 0.70
#             })
        except:
            pass

        try:
            self.correlation_val.to_excel(writer,
                                          sheet_name='Correlaciones_val')
            worksheet_7 = writer.sheets['Correlaciones_val']
        except:
            pass

        writer.save()
        try:
            os.remove('tmp_figure.png')
        except:
            pass

        print("Nombre del archivo:", filename)
        print("Guardado en:", Path(output_route).resolve())
    except:
        raise (ValueError(
            'Error con el Output_route. Verifique que sea correcto el directorio a guardar.'
        ))


def format_excel(gain_table, sheet_name, writer, output_route, filename, p_q,
                 p_gb, benchmark_flag):
    lista_pct = [
        'Total_Dist', 'Total_cum%', 'Total_decum%', 'Good_Dist', 'Good_cum%',
        'Good_decum%', 'Bad_Dist', 'Bad_cum%', 'Bad_decum%', 'Bad_Rate',
        'Bad_Rate_cum%', 'Bad_Rate_decum%'
    ]
    gain_table_aux = gain_table.copy()
    #     for i in lista_pct:
    #         gain_table_aux[i] = gain_table_aux[i].str.\
    #             replace('%', '').astype('float')/100
    # Formateamos la salida en el Excel
    set_order = [
        'min_score', 'max_score', 'Total_Q', 'Total_Dist', 'Total_cum%',
        'Total_decum%', 'Good_Q', 'Good_Dist', 'Good_cum%', 'Good_decum%',
        'Bad_Q', 'Bad_Dist', 'Bad_cum%', 'Bad_decum%', 'Bad_Rate',
        'Bad_Rate_cum%', 'Bad_Rate_decum%', 'ks', 'Odds'
    ]

    # seteamos celda a partir de la cual se imprime el gráfico
    print_graf = 'B' + str(p_q + 5)

    # Creamos la tabla ordenada y eliminamos la columna de max_ks
    final_table = pd.DataFrame(gain_table_aux[set_order])

    # Convert the dataframe to an XlsxWriter Excel object.
    final_table.to_excel(writer, sheet_name=sheet_name)

    # Get the xlsxwriter workbook and worksheet objects.
    workbook = writer.book
    worksheet = writer.sheets[sheet_name]

    # Ocultamos las lineas de cuadricula
    worksheet.hide_gridlines(option=2)
    if not benchmark_flag:
        if isinstance(output_route, str):
            # Insert the chart into the worksheet.
            worksheet.insert_image(print_graf,
                                   output_route + filename + '.png', {
                                       'x_scale': 0.70,
                                       'y_scale': 0.70
                                   })
        else:
            # Insert the chart into the worksheet.
            worksheet.insert_image(print_graf,
                                   output_route / (filename + '.png'), {
                                       'x_scale': 0.70,
                                       'y_scale': 0.70
                                   })

    # Estiramos un poco el ancho de las columnas
    worksheet.set_column('A:P', 11)

    # Alineamos porcentajes y nros a la derecha y dejamos centrados los
    # score.
    alineado_derecha = workbook.add_format({'align': 'right'})
    alineado_pct = workbook.add_format({
        'align': 'right',
        'num_format': '0.00%',
    })
    alineado_centrado = workbook.add_format({'align': 'center'})
    worksheet.set_column('D:D', 11, alineado_derecha)
    worksheet.set_column('E:G', 11, alineado_pct)
    worksheet.set_column('H:H', 11, alineado_derecha)
    worksheet.set_column('I:K', 11, alineado_pct)
    worksheet.set_column('L:L', 11, alineado_derecha)
    worksheet.set_column('M:S', 11, alineado_pct)
    worksheet.set_column('S:T', 11, alineado_centrado)

    #Formateamos la columna Odds
    format_odds_ratio = workbook.add_format({'bold': True})

    format_odds = 'T2:T' + str(p_q + 1)
    worksheet.conditional_format(
        format_odds, {
            'type': '3_color_scale',
            'min_color': "#FF0000",
            'mid_color': "#FFFF00",
            'max_color': "#008000"
        })

    worksheet.set_column('T:T', 11, format_odds_ratio)

    # Creamos el formato de los titulos de las columnas
    header_format = workbook.add_format({
        'bold': True,
        'text_wrap': True,
        'align': 'left',
        'bg_color': "#9C0006",
        'font_color': 'white',
        'border': 1
    })

    #Aplcamos el formato (se aplica solo el color de las letras y el
    # fondo de la celda.)
    worksheet.conditional_format('A1:T1', {
        'type': 'no_blanks',
        'format': header_format
    })

    #Aplicamos "text_wrap" para que se vea el nombre completo de las celdas
    for col_num, value in enumerate(final_table.columns.values):
        worksheet.write(
            0, col_num + 1, value,
            workbook.add_format({
                'valign': 'vcenter',
                'text_wrap': True
            }))

    worksheet.write(
        0, 0, gain_table_aux.index.name,
        workbook.add_format({
            'valign': 'vcenter',
            'text_wrap': True
        }))

    #Creamos el formato de los bordes de celdas
    border_format = workbook.add_format({'border': 1})
    #Aplicamos el formato a la tabla
    format_rag = 'A1:T' + str(p_q + 1)
    worksheet.conditional_format(format_rag, {
        'type': 'no_blanks',
        'format': border_format
    })

    # Le damos formato condicional al max_ks para que quede resaltado.
    # El max_ks lo obtenemos con np.

    format1 = workbook.add_format({
        'bg_color': '#FFC7CE',
        'font_color': '#9C0006',
        'bold': True
    })

    max_ks = np.max(final_table.ks)

    # Rango de Formato condicional de KS
    format_ks = 'S2:S' + str(p_q + 1)
    worksheet.conditional_format(format_ks, {
        'type': 'cell',
        'criteria': '>=',
        'value': max_ks,
        'format': format1
    })

    format_bad = 'P2:P' + str(p_q + 1)
    worksheet.conditional_format(
        format_bad, {
            'type': '3_color_scale',
            'min_color': "#008000",
            'mid_color': "#FFFF00",
            'max_color': "#FF0000"
        })
    #Formateamos la columna bad rate
    format_bad_rate = workbook.add_format({
        'bold': True,
        'num_format': '0.00%',
        'align': 'right'
    })
    worksheet.set_column(format_bad, 11, format_bad_rate)

    # Formateamos las variables de numero para que tengan la marca de miles.

    formato_miles = workbook.add_format({'num_format': '#,##0'})

    def asigna_miles(celda):
        worksheet.set_column(celda, 11, formato_miles)

    asigna_miles(celda='D:D')
    asigna_miles(celda='H:H')
    asigna_miles(celda='L:L')

# Incorporamos colores para poder imprimir de forma más legible
# (esto sirve para la función print(color.ALGUNCOLOR + "texto" + color.END))
class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    GREY = '\033[90m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


