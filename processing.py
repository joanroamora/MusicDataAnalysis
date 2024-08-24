import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from pandas.plotting import scatter_matrix, parallel_coordinates, andrews_curves

# Asegúrate de que la carpeta ./outputs existe
if not os.path.exists('./outputs'):
    os.makedirs('./outputs')

# Carga un dataset desde una ruta especificada
def load_dataset(path):
    return pd.read_csv(path)

# Verifica si hay valores nulos en el DataFrame
def check_null_values(df):
    return df.isnull().values.any()

# Genera un gráfico de caja (boxplot) inicial para visualizar la distribución de los datos
def plot_initial_boxplot(df):
    plt.figure(figsize=(20,8))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.savefig('./outputs/initial_boxplot.png')
    plt.close()

# Escala las características del DataFrame utilizando estandarización (media=0, desviación estándar=1)
def scale_features(df):
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df.values)
    return pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

# Filtra los outliers del DataFrame utilizando la desviación estándar
def filter_outliers(df):
    return df[df.apply(lambda x: np.abs(x - x.mean()) / x.std() < 6).all(axis=1)]

# Genera un gráfico de caja (boxplot) para los datos filtrados
def plot_filtered_boxplot(df):
    plt.figure(figsize=(14,8))
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('./outputs/filtered_boxplot.png')
    plt.close()

# Genera una matriz de dispersión (scatter matrix) de las características seleccionadas
def plot_scatter_matrix(df, feature_columns, target_column):
    color_function = lambda x: 'red' if x > 50 else 'blue'
    colors = df[target_column].map(color_function)
    
    sm = scatter_matrix(df[feature_columns], c=colors, alpha=0.5, figsize=(15, 15))
    
    [s.xaxis.label.set_rotation(45) for s in sm.reshape(-1)]
    [s.yaxis.label.set_rotation(0) for s in sm.reshape(-1)]
    [s.get_yaxis().set_label_coords(-0.9,0.5) for s in sm.reshape(-1)]
    [s.set_xticks(()) for s in sm.reshape(-1)]
    [s.set_yticks(()) for s in sm.reshape(-1)]
    plt.savefig('./outputs/scatter_matrix.png')
    plt.close()

# Genera un heatmap de correlación entre las características del DataFrame
def plot_correlation_heatmap(df):
    corrMat = df.corr()
    
    plt.figure(figsize=(20,20))
    ax = sns.heatmap(corrMat, vmax=1, vmin=-1, cbar_kws={"shrink": .8}, square=True, annot=True, fmt='.2f', cmap='GnBu', center=0)
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    plt.savefig('./outputs/correlation_heatmap.png')
    plt.close()

# Genera un gráfico de coordenadas paralelas para visualizar las relaciones entre las características y la variable objetivo
def plot_parallel_coordinates(df, target_column):
    parallel_coordinates(df, target_column, colormap='cool', xticks=None)
    plt.savefig('./outputs/parallel_coordinates.png')
    plt.close()

# Genera curvas de Andrews para visualizar la distribución de las características en función de la variable objetivo
def plot_andrews_curves(df, target_column):
    andrews_curves(df, target_column, colormap='rainbow')
    plt.savefig('./outputs/andrews_curves.png')
    plt.close()
