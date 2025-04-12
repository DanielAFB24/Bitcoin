import os
import re
import string
import requests
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
from gensim import corpora, models
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
from spacy import displacy



class DataExtractor:
    def __init__(self, source: str = None, chunksize: int = 100000):
        """
        Inicializa el extractor de datos.
        """
        self.source = source
        self.data = None
        self.chunksize = chunksize

    def load_data_api(self, query: str = "#tesla", max_results: int = 200,
                      output_file: str = "tweets_from_api.csv") -> pd.DataFrame:
        from dotenv import load_dotenv
        import json
        import time

        load_dotenv()

        url = "https://twitter-api45.p.rapidapi.com/search.php"
        headers = {
            "X-RapidAPI-Key": os.getenv("RAPIDAPI_KEY"),
            "X-RapidAPI-Host": "twitter-api45.p.rapidapi.com"
        }

        tweets = []
        next_cursor = None
        page_limit = 100
        total_fetched = 0

        while total_fetched < max_results:
            params = {
                "query": query,
                "limit": str(min(page_limit, max_results - total_fetched))
            }
            if next_cursor:
                params["cursor"] = next_cursor

            response = requests.get(url, headers=headers, params=params)
            print(f"🟡 Código: {response.status_code} | Total descargados: {total_fetched}")

            if response.status_code != 200:
                print(f" Error {response.status_code}: {response.text}")
                break

            json_data = response.json()
            page_tweets = json_data.get("timeline", [])
            if not page_tweets:
                print(" No se encontraron más tweets.")
                break

            tweets.extend(page_tweets)
            total_fetched += len(page_tweets)
            next_cursor = json_data.get("next_cursor")

            if not next_cursor:
                print("🔚 No hay más páginas disponibles.")
                break

            time.sleep(1)  # Respetar límite de uso

        print(f" Se extrajeron {len(tweets)} tweets.")

        data = []
        for tweet in tweets:
            user_info = tweet.get("user_info", {})
            entities = tweet.get("entities", {})
            hashtags = [h["text"] for h in entities.get("hashtags", [])]

            data.append({
                "user_name": tweet.get("screen_name", ""),
                "user_location": user_info.get("location", ""),
                "user_description": user_info.get("description", ""),
                "user_created": "",  # no disponible
                "user_followers": user_info.get("followers_count", 0),
                "user_friends": user_info.get("friends_count", 0),
                "user_favourites": user_info.get("favourites_count", 0),
                "user_verified": user_info.get("verified", False),
                "date": tweet.get("created_at", ""),
                "text": tweet.get("text", ""),
                "hashtags": hashtags,
                "source": tweet.get("source", ""),
                "is_retweet": False
            })

        df = pd.DataFrame(data)
        df.to_csv(output_file, index=False, encoding="utf-8")
        self.data = df
        self.source = output_file

        print(f"Archivo '{output_file}' guardado con {len(df)} filas.")
        return df

    def load_data(self) -> pd.DataFrame:
        """
        Carga los datos desde el archivo especificado en self.source si aún no se han cargado.

        Devuelve:
        DataFrame con los datos cargados.
        """
        if self.data is None and self.source is not None:
            try:
                self.data = pd.read_csv(self.source)
                print(f" Datos cargados desde archivo '{self.source}'.")
            except Exception as e:
                print(f" Error al cargar el archivo: {e}")
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
        Realiza un análisis avanzado de hashtags sobre el DataFrame en self.data.
        """
        if self.data is not None:
            data = self.data.copy()

            # 1. Normaliza el texto
            data['text'] = data['text'].apply(self.clean_text)

            # 2. Extrae los hashtags
            data['Only_hashtag'] = data['text'].apply(self.extract_hashtags)

            # 3. Convierte la fecha
            data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.date

            # 4. Explota los hashtags
            data_exploded = data.explode('Only_hashtag')
            data_exploded = data_exploded[data_exploded['Only_hashtag'].notnull()]

            # 5. Análisis
            overall = (
                data_exploded['Only_hashtag']
                .value_counts()
                .reset_index()
                .rename(columns={'index': 'hashtag', 'Only_hashtag': 'frequency'})
            )

            by_user = (
                data_exploded.groupby(['user_name', 'Only_hashtag'])
                .size()
                .reset_index(name='frequency')
                .rename(columns={'Only_hashtag': 'hashtag'})
                .sort_values(by='frequency', ascending=False)
            )

            by_date = (
                data_exploded.groupby(['date', 'Only_hashtag'])
                .size()
                .reset_index(name='frequency')
                .rename(columns={'Only_hashtag': 'hashtag'})
                .sort_values(by=['date', 'frequency'], ascending=[True, False])
            )

            analytics = {
                'overall': overall,
                'by_user': by_user,
                'by_date': by_date,
                'data': data_exploded
            }

            print(f'Número total de hashtags: {len(overall)}')
            return analytics

        else:
            print(" No hay datos cargados. Usa 'load_data_api()' primero.")
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


    def model_topics(self, num_topics: int = 5, passes: int = 10) -> list:
        if self.data is None or 'text' not in self.data.columns:
            print(" No hay datos. Carga los datos primero.")
            return []

        # Limpieza previa básica si no existe clean_text
        self.data['clean_text'] = self.data['text'].apply(self.clean_text)

        stop_words = set(stopwords.words("english") + list(string.punctuation))

        # Filtrar stopwords
        texts = [
            [word for word in t.split() if word not in stop_words]
            for t in self.data['clean_text'] if isinstance(t, str)
        ]

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]

        lda_model = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=passes)

        topics = lda_model.print_topics(num_words=6)
        for i, topic in enumerate(topics):
            print(f"\n Tópico {i + 1}:")
            palabras = topic[1].split(' + ')
            for palabra in palabras:
                peso, palabra_clave = palabra.split('*')
                print(f" - {palabra_clave.replace('\"', '')}: {float(peso):.3f}")

        return topics


    def analyze_sentiment(self, method: str = 'textblob') -> pd.DataFrame:
        if self.data is None or 'text' not in self.data.columns:
            print(" No hay datos cargados.")
            return pd.DataFrame()

        self.data['clean_text'] = self.data['text'].apply(self.clean_text)

        if method == 'textblob':
            self.data['sentiment_polarity'] = self.data['clean_text'].apply(lambda t: TextBlob(t).sentiment.polarity)
            self.data['sentiment_subjectivity'] = self.data['clean_text'].apply(
                lambda t: TextBlob(t).sentiment.subjectivity)
        else:


            nlp = spacy.load("en_core_web_sm")
            nlp.add_pipe("spacytextblob")
            self.data['sentiment_polarity'] = self.data['clean_text'].apply(lambda t: nlp(t)._.polarity)
            self.data['sentiment_subjectivity'] = self.data['clean_text'].apply(lambda t: nlp(t)._.subjectivity)

        print(" Análisis de sentimiento completado.")
        return self.data


    def parse_and_summarize(self, summary_ratio: float = 0.3) -> str:
        if self.data is None or 'text' not in self.data.columns:
            print(" No hay datos.")
            return ""

        text = " ".join(self.data['text'].dropna())
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())

        stop_words = set(stopwords.words("english") + list(string.punctuation))
        words_filtered = [w for w in words if w not in stop_words]

        word_freq = Counter(words_filtered)
        sentence_scores = {sent: sum(word_freq[w] for w in word_tokenize(sent.lower()) if w in word_freq)
                           for sent in sentences}

        n = max(1, int(len(sentences) * summary_ratio))
        top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:n]

        summary = " ".join([s for s in sentences if s in top_sentences])
        print("\n Resumen generado:")
        print(summary)
        return summary


    def plot_sentiment_distribution(self, output_path: str = "sentiment_distribution.png"):
        if self.data is None or 'sentiment_polarity' not in self.data.columns:
            print(" Asegúrate de haber ejecutado el análisis de sentimiento primero.")
            return

        plt.figure(figsize=(8, 5))
        plt.hist(self.data['sentiment_polarity'], bins=30, color='skyblue', edgecolor='black')
        plt.title('Distribución de Sentimiento (Polaridad)')
        plt.xlabel('Polaridad')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        print(f" Gráfico de sentimiento guardado como '{output_path}'")
        plt.close()

        # Agregar tabla resumen
        neg = len(self.data[self.data['sentiment_polarity'] < 0])
        neu = len(self.data[self.data['sentiment_polarity'] == 0])
        pos = len(self.data[self.data['sentiment_polarity'] > 0])
        print("\n Tabla de resumen de sentimiento:")
        print(pd.DataFrame({
            'Sentimiento': ['Negativo', 'Neutro', 'Positivo'],
            'Cantidad': [neg, neu, pos]
        }))


    def visualize_dependency_tree(self, index: int = 0, output_html: str = "dependency_tree.html"):
        if self.data is None or 'clean_text' not in self.data.columns:
            print(" No hay datos procesados. Ejecuta primero analyze_sentiment().")
            return

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.data['clean_text'].iloc[index])

        print(f"\n Árbol sintáctico de la oración {index} guardado en {output_html}")

        svg = displacy.render(doc, style="dep", jupyter=False)
        with open(output_html, "w", encoding="utf-8") as f:
            f.write(svg)

        print(f" Puedes abrir el archivo HTML generado para ver el árbol sintáctico.")
