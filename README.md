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

# Documentación de Técnicas y Resultados parte II

# Twitter Advanced Text Analysis

Este proyecto implementa un análisis avanzado de textos extraídos desde la API de Twitter a través de RapidAPI. Incluye extracción de datos, preprocesamiento, modelado de tópicos (LDA), análisis de sentimientos y visualización de estructuras sintácticas.

## 📌 Fuente de datos

Se utiliza la API "Twitter API v2" disponible en [RapidAPI](https://rapidapi.com/alexanderxbx/api/twitter-api45). Esta API permite realizar búsquedas de tweets utilizando parámetros personalizados como:

- `query`: palabra clave o hashtag a buscar (por ejemplo, `#Tesla`)
- `limit`: número máximo de tweets (ej. 200)
- `start`: fecha de inicio (ej. `2025-04-01`)

---

## 🧠 Metodología

### 1. Extracción de Datos

Se conecta con la API usando la librería `requests` y se configuran los headers con claves seguras a través del archivo `.env`.

```python
url = "https://twitter-api45.p.rapidapi.com/search.php"
headers = {
    "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
    "X-RapidAPI-Host": "twitter-api45.p.rapidapi.com"
}
```

## Modelado de Tópicos con LDA (Latent Dirichlet Allocation)
- Técnica Aplicada: Se aplicó LDA con la librería gensim para descubrir temas (tópicos) dominantes en los tweets. El texto fue previamente limpiado y tokenizado, y se eliminaron stopwords para mejorar la calidad de los temas.
- Justificación: LDA permite encontrar grupos de palabras que tienden a aparecer juntas, lo cual revela temas latentes en el corpus de textos. Es muy útil en análisis exploratorio de contenido.

Resultado: Se imprimieron 5 tópicos. Cada tópico contiene palabras clave con sus pesos. Ejemplo:

🧠 Tópico 1:
 - "#tesla": 0.051
 - "#": 0.039
 - "you": 0.014
 - "tesla": 0.013
 - "tsla": 0.012
 - "model": 0.008

## Análisis de Sentimiento (TextBlob y spaCy)
- Técnica Aplicada: Se utilizó TextBlob (y opcionalmente spaCyTextBlob) para calcular la polaridad y subjetividad de cada texto. Estos valores se guardaron en las columnas sentiment_polarity y sentiment_subjectivity.

- Justificación: Ayuda a identificar la actitud (positiva/negativa) de los usuarios hacia los temas comentados.

- Resultado: Se imprimió un mensaje indicando la finalización del análisis. Además, se visualiza la distribución de polaridad con un histograma:

## Gráfico de Distribución de Sentimientos:

Se observa la frecuencia de tweets negativos, neutros y positivos.

## Parsing y Árboles Sintácticos
- écnica Aplicada: Con spaCy, se cargó un modelo de procesamiento de lenguaje para visualizar la estructura gramatical de una oración mediante árboles de dependencias (dependency trees).

- Justificación: Esto permite entender cómo se relacionan las palabras dentro de una frase, útil para análisis sintáctico profundo.

- Resultado: Se muestra visualmente un árbol sintáctico en navegador. Cada palabra conecta con su “palabra padre”, reflejando la estructura de la oración.

## Resumen Extractivo (Método Tradicional con Frecuencia de Palabras)
- Técnica Aplicada: Se utilizó nltk para dividir el texto en oraciones, calcular la frecuencia de palabras y seleccionar las más representativas para formar un resumen.

- Justificación: Permite sintetizar los puntos clave del corpus de forma automática.

- Resultado:  Resumen generado: Se imprime en consola una selección de frases relevantes, representativas del contenido total del corpus.

## Intrucciones para reproducir

```
pip install -r requirements.txt
```

Crear archivo .env con tu clave de RapidAPI:

```.env
RAPIDAPI_KEY=tu_clave_aqui
```

ejecutar python main.py.

El proyecto genera :
- Gráfico de sentimiento guardado como 'sentiment_distribution.png'
- Tabla de resumen de sentimiento (Por consola)
- Árbol sintáctico con una de las oraciones guardado en dependency_tree.html
- Resumen de los comentarios mas destacados


#  Interacción y Análisis de Redes en Twitter


## 2. Construcción del Grafo de Interacciones

### Método: `build_interaction_graph`

```python
def build_interaction_graph(self) -> nx.DiGraph
```

Construye un grafo dirigido en el que los nodos representan usuarios, y las aristas indican menciones de un usuario a otro dentro del texto.

####  Requisitos
- `self.data`: debe contener las columnas `user_name` y `text`.
- Los textos se normalizan (minúsculas y sin espacios).

####  Lógica
- Se filtran textos vacíos o nulos.
- Se extraen menciones con regex: `@(\w{1,15})`.
- Se añade un nodo por cada usuario.
- Se crea una arista desde el usuario emisor hacia el usuario mencionado.

####  Ejemplo de uso:
```python
grafo = extractor.build_interaction_graph()
```

####  Resultado:
- Grafo dirigido (`nx.DiGraph`) guardado también en `self.graph`.
- Mensaje en consola con el número de nodos y aristas.

---

## 3. Análisis de la Red

### Método: `analyze_network`

```python
def analyze_network(self, G: nx.DiGraph)
```

Analiza el grafo generado con métricas de red y detecta comunidades con Louvain. También genera una visualización básica.

#### ✏️ Estadísticas Calculadas:
-  **Top 5 usuarios que más mencionan** (grado de salida).
-  **Top 5 usuarios más mencionados** (grado de entrada).
-  **Top 5 en centralidad de intermediación** (usuarios puente).
-  **Top 5 en centralidad de cercanía** (usuarios accesibles).
-  **Top 5 en centralidad de autovector** (usuarios con prestigio estructural).
-  **Detección de comunidades** con Louvain (requiere red no dirigida).

####  Visualización:
Se genera un grafo con `matplotlib` usando `spring_layout` y se dibujan nodos y aristas con bajo nivel de detalle.

####  Resultados Guardados:
- `self.top_out_degree`
- `self.top_in_degree`
- `self.top_betweenness`
- `self.top_closeness`
- `self.top_eigenvector`

---

## 4. Chat con Modelo de Lenguaje Local

### Método: `chat_local_llm`

```python
def chat_local_llm(self, prompt: str = None)
```

Carga e interactúa con un modelo de lenguaje local (`google/gemma-2-2b-it`) en consola.

####  Flujo:
- Carga el modelo y el tokenizador desde HuggingFace.
- Usa GPU si está disponible.
- Permite ingresar un `prompt` o interactuar dinámicamente por consola.
- Imprime la respuesta generada por el modelo.

---

## 5. Generación de Prompts desde la Red

### Método: `generate_prompt_from_network`

```python
def generate_prompt_from_network(self, G: nx.DiGraph) -> str
```

Genera un prompt en lenguaje natural basado en los análisis de red para alimentar un modelo LLM.

####  Detalles:
- Usa el top 3 de usuarios con mayor grado de salida.
- Usa el hashtag más frecuente extraído del análisis extendido.
- Devuelve un string con una pregunta interpretativa lista para ser usada con un LLM.

####  Ejemplo de Prompt:

```
Se ha analizado una red de interacciones en Twitter basada en menciones...

Los 3 usuarios más activos...
El hashtag más frecuente...
¿Qué factores podrían explicar...
```

---

