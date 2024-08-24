import processing as proc

def main():
    # Cargar el dataset
    dfMusic = proc.load_dataset('./data/music.csv')
    
    # Verificar valores nulos
    has_nulls = proc.check_null_values(dfMusic)
    print(f"El dataset tiene valores nulos: {has_nulls}")
    
    # Plot inicial del boxplot
    proc.plot_initial_boxplot(dfMusic)
    
    # Escalar características
    dfMusic_ft = dfMusic[dfMusic.columns[1:]]  # Excluyendo la primera columna que es el Track
    dfMusic_ftsc = proc.scale_features(dfMusic_ft)
    
    # Filtrar outliers
    dfMusic_ftscClc = proc.filter_outliers(dfMusic_ftsc)
    print(f"Dimensiones después de filtrar outliers: {dfMusic_ftscClc.shape}")
    
    # Plot del boxplot filtrado
    proc.plot_filtered_boxplot(dfMusic_ftscClc)
    
    # Matriz de dispersión
    features = list(dfMusic.columns[1:11])  # Cambiar según las características que te interesen
    proc.plot_scatter_matrix(dfMusic, features, "Popularity")
    
    # Heatmap de correlación
    proc.plot_correlation_heatmap(dfMusic_ftscClc)
    
    # Coordenadas paralelas
    proc.plot_parallel_coordinates(dfMusic_ftscClc.iloc[:, :150], "Popularity")

    # Curvas de Andrews
    proc.plot_andrews_curves(dfMusic_ftscClc, "Popularity")

if __name__ == "__main__":
    main()
