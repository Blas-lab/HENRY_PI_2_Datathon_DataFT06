# HENRY_PI_2_Datathon_DataFT06

## Datathon

# Descripción:
Este proyecto tiene como objetivo predecir el precio de una propiedad en función de sus características. Se utiliza un modelo de clasificación de árboles de decisión para entrenar con los datos de entrenamiento y hacer predicciones en los datos de prueba.

El archivo "train_clean.csv" contiene los datos de entrenamiento y "test_clean.csv" contiene los datos de prueba.

Se utilizó la biblioteca scikit-learn para construir el modelo y se utilizaron las siguientes funciones:

- Pandas para cargar y manipular los datos
- train_test_split para dividir los datos en conjuntos de entrenamiento y prueba
- DecisionTreeClassifier para construir el modelo de árboles de decisión
- GridSearchCV para encontrar los mejores parámetros para el modelo
- SimpleImputer para manejar valores faltantes en los datos
- accuracy_score para evaluar la precisión del modelo
- Además, se combinaron las columnas "cats_allowed" y "dogs_allowed" en una nueva característica llamada "pets_allowed".

Finalmente, se guardaron las predicciones en un archivo CSV llamado "test_dtc_prediction.csv" y se exportó el conjunto de datos final en un archivo llamado "final_format.csv" sin el nombre de la columna.

Para ejecutar este código, es necesario tener instalado Python y las bibliotecas mencionadas anteriormente. También es necesario tener acceso a los archivos "train_clean.csv" y "test_clean.csv".

# Pipeline:
El proyecto consta de varios pasos críticos en su pipeline de procesamiento de datos. A continuación se detallan estos pasos:

1. Limpieza de datos: En esta etapa, se limpian los datos de entrenamiento y de prueba. Esto incluye eliminar columnas no deseadas, manejar valores faltantes y llevar a cabo la codificación necesaria para las variables categóricas.

2. Imputación de datos: Una vez que se han limpiado los datos, se utiliza un objeto SimpleImputer para rellenar los valores faltantes en las columnas del conjunto de datos de entrenamiento. Este objeto se ajusta a los datos de entrenamiento y luego se utiliza para transformar los datos de prueba.

3. Creación de una nueva característica: Se combinan las columnas "cats_allowed" y "dogs_allowed" en una nueva característica llamada "pets_allowed".

4. Detección de valores atípicos: Se detectan y manejan los valores atípicos en las columnas sqfeet, beds, baths, region_encoded, type_encoded, laundry_option, parking_option y state_encoded.

5. Entrenamiento del modelo: Se entrena un modelo de Decision Tree Classifier utilizando los datos de entrenamiento limpios e imputados.

6. Evaluación del modelo: Se evalúa el rendimiento del modelo entrenado en el conjunto de datos de prueba utilizando métricas como la precisión y la matriz de confusión.

7. Hacer predicciones: Se hacen predicciones utilizando el modelo entrenado en el conjunto de datos de prueba y se exportan a un archivo CSV.

8. Formateo final: Se abre el archivo CSV de predicciones, se crea un nuevo DataFrame que contiene solo la columna "price_scale_predicted", y se exporta a un nuevo archivo CSV llamado "final_format.csv" sin el nombre de la columna.

En resumen, el pipeline del proyecto se centra en limpiar y preparar los datos, entrenar y evaluar un modelo de Decision Tree Classifier, hacer predicciones y exportar los resultados en un formato adecuado.

# Detalle de uso:
Por inconvenientes con el tamaño de archivos y carpetas, solo se puede disponer en github de aquello justo y necesario para correr el proyeto
considerando que la data fatante osea los archivos originales en formato parquet no estan debe crearse una carpeta en el mismo directorio del proyecto con los mismos
## Esta carpeta debe llamarse 'Dataset_parket' una vez hecho estos al correr el archivo 'EDA.py' o el archivo 'EDA.ipynb' todos los demas directorios y archivos necesarios seran creados automaticamente.

# Sobre EDA en el proyecto:
El análisis exploratorio de los datos (EDA, por sus siglas en inglés) es una etapa crucial en cualquier proyecto de análisis de datos. En este proyecto, el EDA se enfocó en comprender el conjunto de datos que se utilizará para el entrenamiento del modelo. Esto incluyó la revisión de las columnas del conjunto de datos, el tipo de datos que contenían, la presencia de valores faltantes y la distribución de los datos.

Una vez que se comprendió el conjunto de datos, se procedió a limpiar los datos para asegurar que el modelo entrenado con ellos fuera lo más preciso posible. Esto incluyó la eliminación de columnas no deseadas, la imputación de valores faltantes y la codificación de variables categóricas.

Además de la limpieza de datos, el EDA también incluyó la generación de gráficos y tablas para visualizar la relación entre las diferentes variables del conjunto de datos. Esto ayudó a identificar patrones y tendencias en los datos, lo que ayudó a informar la selección de variables y la creación del modelo.

En resumen, el EDA en este proyecto se enfocó en comprender y limpiar el conjunto de datos para asegurar que el modelo entrenado con ellos fuera lo más preciso posible. También se utilizaron gráf.

## Algunos de los puntos clave que se abordaron en la etapa de EDA incluyen:
- Análisis de outliers: Se utilizaron diferentes técnicas para detectar y tratar los outliers en las columnas relevantes como sqfeet, beds, baths, region_encoded, type_encoded, laundry_option, parking_option y state_encoded.

- Columnas combinadas para nuevas características: Se combinaron algunas columnas para crear nuevas características relevantes como pets_allowed que combina las columnas cats_allowed y dogs_allowed.

- Transformación de variables cualitativas a cuantitativas: Algunas variables cualitativas como region_encoded, type_encoded, laundry_option, parking_option y state_encoded fueron transformadas en variables cuantitativas mediante el uso de técnicas de codificación como OneHotEncoding.

# Sobre el Modelo Decision Tree Classifier:

El modelo de árbol de decisiones (DTC) es un algoritmo de aprendizaje automático supervisado que se utiliza para resolver problemas de clasificación y regresión. A partir de un conjunto de datos de entrenamiento, el DTC construye un árbol de decisiones que se utiliza para hacer predicciones sobre nuevos datos.

El árbol de decisiones está compuesto por nodos de decisión y hojas de clasificación. Los nodos de decisión representan una pregunta o una condición sobre los datos de entrada, mientras que las hojas representan las predicciones o las clases. El proceso de construcción del árbol se basa en la selección de la mejor pregunta o condición para dividir el conjunto de datos en subsets más homogéneos, con el objetivo de maximizar la información ganada en cada paso.

Una de las ventajas del DTC es que es fácil de entender y explicar, ya que el árbol de decisiones es una representación visual del proceso de toma de decisiones. Sin embargo, también tiene algunas desventajas, como la tendencia a sobreajustar el conjunto de entrenamiento, lo que puede llevar a un bajo rendimiento en conjuntos de datos nuevos. También puede ser sensible a pequeñas variaciones en los datos de entrada.

En este proyecto, se utilizó un modelo DTC para predecir el precio de las propiedades en base a un conjunto de características del conjunto de entrenamiento. Se utilizaron técnicas de preprocesamiento, como la imputación de datos faltantes y la eliminación de outliers, para mejorar la precisión del modelo. Los resultados se compararon con un conjunto de datos de prueba y se exportaron a un archivo CSV para su uso posterior.

# Conclusiones:
En conclusión, este proyecto tuvo como objetivo principal el desarrollar un modelo predictivo para predecir el precio de las propiedades en Estados Unidos. Para lograr esto, se llevó a cabo una serie de tareas que incluyeron la limpieza y preprocesamiento de los datos, la exploración y análisis de los mismos, la creación de nuevas características y la selección de características relevantes.

En cuanto al análisis exploratorio de los datos (EDA), se encontró que hay una gran cantidad de valores atípicos o "outliers" en varias columnas, lo cual es importante tener en cuenta al momento de entrenar el modelo. También se combinaron algunas columnas para crear nuevas características, y se transformaron variables cualitativas en cuantitativas para poder incluirlas en el modelo.

Se utilizó un árbol de decisión como modelo predictivo y se obtuvo una precisión del 80% en las predicciones realizadas on premise, lo cual es un buen resultado teniendo en cuenta la complejidad del problema. Desgraciadamente al llevar las mismas predicciones al Dashboard y compararlas con la columna final de referencia el accurasy fue en el mejor push de 50%.

En general, este proyecto ha demostrado que es posible desarrollar un modelo predictivo preciso pero el nivel de tunnig que hubiera sido necesario para una mejor performance no pudo ser llevado a cabo en el tiempo de la competencia.
De todas maneras el dev del proyecto seguira haciendo cambios a futuros y tratara de incorporar algun modelo no supoervisado como K-means lo cual era otra posibilidad para el Datathon.
