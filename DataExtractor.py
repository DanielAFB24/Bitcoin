import pandas as pd
import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

class DataExtractor:
    def __init__(self, source_file: str, chunksize: int = 100000):

        self.source_file = source_file
        self.data = None
        self.chunksize = chunksize

    def load_data(self):

        try:
            self.data = pd.read_csv(
                self.source_file,
                chunksize=self.chunksize,
                sep=',',
                on_bad_lines='skip',
                engine='python',
                encoding='utf-8'
            )

            print("Datos cargados en el archivo CSV.")

        except Exception as e:
            print(f"Error cargar datos {e}")




        return self.data

    def clean_text(self, texto : str) -> str:
        """
         Limpia y normaliza el texto.
         Pasos sugeridos:
         - Convertir a minúsculas.
         - Eliminar o extraer URLs.
         - Eliminar caracteres especiales (OJO! necesitamos hashtags) y espacios redundantes.
         Devuelve:
         El texto limpio.
         """

        if self.data is not None:
            if isinstance(texto, str):
                texto = texto.lower()  # Convierte a minúsculas
                texto = re.sub(r'http[s]?://\S+|www\.\S+', '', texto)  # Elimina URLs
                texto = re.sub(r'[^a-zA-Z0-9\s#]', '', texto)  # Elimina caracteres especiales, mantiene #
                texto = re.sub(r'\s+', ' ', texto).strip()
                texto = re.sub(r'\.{2,}', '.', texto)
                return texto

        return ""

    def extract_hashtags(self, text: str) -> list:
        """
        Extrae y devuelve una lista de hashtags presentes en el texto.
        Implementación sugerida:
        - Utilizar expresiones regulares para encontrar palabras que comiencen con '#' .
        """
        if isinstance(text, str) and text:
            # Usa findall para extraer todas las palabras que comienzan con '#'
            hashtags = re.findall(r'#\w+', text)
            return hashtags
        return []

    def analytics_hashtags_extended(self) -> dict:
        """
        Realiza un análisis avanzado de hashtags sobre el conjunto de datos cargado (self.data).

        El método realiza los siguientes pasos:
        1. Aplica la función clean_text a la columna 'text' para normalizar los datos.
        2. Extrae los hashtags de cada texto usando extract_hashtags y los almacena en una nueva columna.
        3. Convierte la columna 'date' a tipo datetime y extrae solo la fecha (sin la hora).
        4. Explota la columna de hashtags para obtener una fila por cada hashtag, lo que facilita los cálculos de
       frecuencia.

        5. Calcula tres análisis:
        - Frecuencia total de cada hashtag (overall).
        - Frecuencia de hashtags por usuario (by_user).
        - Evolución de la frecuencia de hashtags por día (by_date).

        Retorna:
        Un diccionario con tres DataFrames, con claves:
        'overall': DataFrame con columnas ['hashtag', 'frequency'].
        'by_user': DataFrame con columnas ['user_name', 'hashtag', 'frequency'].
        'by_date': DataFrame con columnas ['date', 'hashtag', 'frequency'].
        """
        if self.data is not None:
            processed_chunks = []  # Almacena los chunks procesados

            for chunk in self.data:
                # 1. Normaliza el texto
                chunk['text'] = chunk['text'].apply(self.clean_text)

                # 2. Extrae los hashtags y los almacena en 'Only_hashtag'
                chunk['Only_hashtag'] = chunk['text'].apply(self.extract_hashtags)

                # 3. Convierte la columna 'date' a solo fecha
                chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce').dt.date

                # Almacena el chunk procesado
                processed_chunks.append(chunk)

                # 4. Combina todos los chunks procesados en un solo DataFrame
                data = pd.concat(processed_chunks, ignore_index=True)

                # 5. Explota la columna 'Only_hashtag' para obtener una fila por cada hashtag
                data_exploded = data.explode('Only_hashtag')

                # Filtra filas vacías (sin hashtags)
                data_exploded = data_exploded[data_exploded['Only_hashtag'].notnull()]

                # 🟢 1. Frecuencia total de cada hashtag (overall)
                overall = (
                    data_exploded['Only_hashtag']
                    .value_counts()
                    .reset_index()
                    .rename(columns={'index': 'hashtag', 'Only_hashtag': 'frequency'})
                )

                # 🟢 2. Frecuencia de hashtags por usuario (by_user)
                by_user = (
                    data_exploded.groupby(['user_name', 'Only_hashtag'])
                    .size()
                    .reset_index(name='frequency')
                    .rename(columns={'Only_hashtag': 'hashtag'})
                    .sort_values(by='frequency', ascending=False)
                )

                # 🟢 3. Evolución de la frecuencia de hashtags por día (by_date)
                by_date = (
                    data_exploded.groupby(['date', 'Only_hashtag'])
                    .size()
                    .reset_index(name='frequency')
                    .rename(columns={'Only_hashtag': 'hashtag'})
                    .sort_values(by=['date', 'frequency'], ascending=[True, False])
                )

                # Preparar el resultado en un diccionario con DataFrames
                analytics = {
                    'overall': overall,
                    'by_user': by_user,
                    'by_date': by_date,
                    'data': data_exploded  # Devuelve el DataFrame completo para análisis adicionales
                }

                print(f'Número total de hashtags: {len(overall)}')
                return analytics

        else:
            print("No hay datos cargados. Usa 'load_data()' primero.")
            return {}

    def generate_hashtag_wordcloud(self, overall_df: pd.DataFrame = None, max_words: int = 100, figsize:
    tuple = (10, 6)) -> None:
        """
        Genera y muestra una wordcloud basada en el análisis global de hashtags.
   
        Este método utiliza el DataFrame 'overall' que contiene la frecuencia global de cada hashtag.
        Si no se proporciona el DataFrame, se calcula llamando a analytics_hashtags_extended().
   
        Parámetros:
        overall_df (pd.DataFrame, opcional): DataFrame con columnas ['hashtags', 'frequency']. Si es None, se
       calcula.
        max_words (int, opcional): Número máximo de palabras a incluir en la wordcloud.
        figsize (tuple, opcional): Tamaño de la figura a mostrar.
   
        Proceso:
        1. Si overall_df es None, llamar a analytics_hashtags_extended y extraer la parte 'overall'.
        2. Convertir el DataFrame a un diccionario donde las claves sean los hashtags y los valores sean las
       frecuencias.
        3. Utilizar la clase WordCloud de la librería wordcloud para generar la nube de palabras.
        4. Visualizar la wordcloud con matplotlib.
        """

        # 1. Cargar el DataFrame 'overall' si no se proporciona
        if overall_df is None:
            overall_df = self.analytics_hashtags_extended().get('overall')

        if overall_df is None or overall_df.empty:
            print("No hay datos disponibles para generar la WordCloud.")
            return

        overall_df = overall_df.rename(columns={'frequency': 'hashtag', 'count': 'frequency'})

        # 2. Convertir el DataFrame a un diccionario adecuado
        hashtag_frequencies = overall_df.set_index('hashtag')['frequency'].to_dict()

        # 3. Generar la nube de palabras
        wordcloud = WordCloud(
            width=800,
            height=600,
            background_color="white",
            max_words=max_words,
            colormap='viridis'
        ).generate_from_frequencies(hashtag_frequencies)

        # 4. Visualizar la nube de palabras
        plt.figure(figsize=figsize)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Nube de Palabras de Hashtags")
        plt.show()
















