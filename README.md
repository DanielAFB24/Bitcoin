# Bitcoin
## 1. Descripción General

La clase `DataExtractor` fue diseñada para:

1. Cargar datos desde un archivo CSV (posiblemente muy grande) en modo por *chunks* (bloques).
2. Limpiar y normalizar el texto, **manteniendo** hashtags.
3. Extraer hashtags de cada texto.
4. Calcular la frecuencia de uso de los hashtags de forma global, por usuario y por fecha.
5. Generar una nube de palabras (WordCloud) a partir de los hashtags más frecuentes.

El objetivo principal es facilitar el análisis de hashtags en un conjunto grande de publicaciones o comentarios de usuarios (p.e. tuits, posts de redes sociales, foros, etc.).

---

## 2. Fuente de Datos Utilizada

- **Formato CSV**: Se espera que el archivo de origen (source_file) sea un CSV con, al menos, las columnas:
  - `text`: El texto publicado por el usuario (puede contener hashtags).
  - `date`: Fecha y hora aproximada de la publicación.
  - `user_name`: El nombre de usuario o identificador de quien publicó el texto.

- **Separador**: Coma (`,`).
- **Encabezados Requeridos**: `user_name`, `text`, `date` (como mínimo).
- **Lectura por Bloques**: Se usa el parámetro `chunksize` para leer el archivo en porciones de 100,000 filas, lo cual ayuda a evitar problemas de memoria en archivos muy grandes.

# 3. Metodología de Extracción, Limpieza y Análisis

## 3.1 Extracción de Datos

### Instancia de la clase:

```python
extractor = DataExtractor(source_file='ruta/al/archivo.csv', chunksize=100000)
```

source_file: Ruta completa al archivo CSV.
chunksize: Cantidad de filas que se leen por bloque (por defecto 100,000).
```python 
extractor.load_data()
```
Se utiliza pandas.read_csv con sep=',', on_bad_lines='skip' y encoding='utf-8'.
Almacena la referencia a los datos en self.data, que será un iterador de chunks de pandas.

## 3.2 Limpieza de Texto
- Se convierte el texto a minúsculas.
- Se eliminan URLs (http, https, www).
- Se eliminan caracteres especiales, excepto #, para preservar los hashtags.
- Se quitan espacios redundantes y secuencias de puntos repetidos.
- La función clean_text realiza estos pasos, devolviendo una cadena de texto limpio.

## 3.3 Extracción de Hashtags
Mediante extract_hashtags, se usa una expresión regular (r'#\w+') para encontrar todas las palabras que inician con #.
Devuelve una lista de hashtags por cada texto.
## 3.4 Análisis Avanzado de Hashtags
El método analytics_hashtags_extended realiza:
- Normalización: Aplica clean_text a cada texto en la columna text.
- Extracción: Crea una columna Only_hashtag con la lista de hashtags encontrados.
- Conversión de Fechas: Convierte la columna date para obtener solo la parte de fecha (YYYY-MM-DD), ignorando la hora.
- Explosión de Hashtags: Usa explode para crear una fila por cada hashtag en Only_hashtag.

### Cálculo de Frecuencias:

- overall: Frecuencia total de cada hashtag.
- by_user: Frecuencia de cada hashtag por usuario.
- by_date: Frecuencia de cada hashtag por día.
- Retorna un diccionario con estos DataFrames, además del data_exploded con toda la información.

# Ejecución 
Entrar al proyecto y ejecutar el comando python main.py. Realizara el procedimiento explicado para posteriormente aparecer la imagen generada por  WordCloud.
