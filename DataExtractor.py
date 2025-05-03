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
import time
import networkx as nx
from networkx.algorithms import community
from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM





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
                "query": query

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

            self.analytics = analytics
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


    def build_interaction_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()

        if self.data is None:
            print(" No hay datos para generar el grado .")
            return G



        df = self.data.dropna(subset=['user_name', 'text'])

        for _ , row in df.iterrows():
            user = row['user_name'].lower().strip()
            text = row['text'].lower().strip()

            if not isinstance(text, str):
                continue

            G.add_node(user)
            menciones = re.findall(r"@(\w{1,15})", text)

            for mencion in menciones:
                G.add_edge(user,mencion)

        print(f"Grafo construido con {G.number_of_nodes()} nodos y {G.number_of_edges()} aristas.")
        self.graph = G
        return G

    def analyze_network(self, G: nx.DiGraph):

        """
         Calcula métricas de red y detecta comunidades utilizando el algoritmo de Louvain.
         Imprime estadísticas y genera una visualización básica.
         """

        print("************************* ANALISIS GRAGO ********************************")

        if G.number_of_nodes() == 0:
            print("El grafo está vacío. Verifica si hay datos cargados.")
            return

        # Calculamos los frados de salida

        out_degs = dict(G.out_degree())

        top_out = sorted(out_degs.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n🔸 Top 5 usuarios que más mencionan a otros (grado de salida):")
        for user, grado in top_out:
            print(f" - {user}: {grado} menciones")

        in_degs = dict(G.in_degree())

        top_in = sorted(in_degs.items(), key=lambda x: x[1], reverse=True)[:5]
        print("\n🔹 Top 5 usuarios más mencionados (grado de entrada):")
        for user, grado in top_in:
            print(f" - {user}: mencionado {grado} veces")

        G_undirected = G.to_undirected()

        # Betweenness Centrality

        bet_centrality = nx.betweenness_centrality(G, k=10, normalized=True)
        top5_node_centrality = sorted(bet_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

        print("\n🎯 Top 5 usuarios puente (centralidad de intermediación):")
        print("   Usuarios que aparecen con más frecuencia en los caminos más cortos entre otros.")
        for user, score in top5_node_centrality:
            print(f" - {user}: {score:.4f}")

        # Closeness Centrality
        closeness_centrality = nx.closeness_centrality(G_undirected)
        top5_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]

        print("\n🎯 Top 5 usuarios más accesibles (centralidad de cercanía):")
        print("   Usuarios que pueden llegar rápidamente a otros usuarios en la red (menor número de saltos promedio).")
        for user, score in top5_closeness:
            print(f" - {user}: {score:.4f}")

        # Calculamos Egevector Centrality
        eigenvector_centrality = nx.eigenvector_centrality(G_undirected, max_iter=1000)
        top5_eigevector = sorted(eigenvector_centrality.items(), key=lambda x: x[1])[:5]

        print("\n🎯 Top 5 usuarios con mayor prestigio estructural (centralidad de autovector):")
        print("   Usuarios conectados a otros usuarios influyentes, según la estructura de la red.")
        for user, score in top5_eigevector:
            print(f" - {user}: {score:.4f}")

        print("**************************** COMUNIDADES ************************")

        try:
            communities_louvain = community.louvain_communities(G_undirected, weight=None, resolution=1.0, seed=None)
            # 'communities_louvain' es una lista de conjuntos (cada conjunto = comunidad)
            print(f"Número de comunidades detectadas: {len(communities_louvain)}")
            print(f"Comunidades (primeras 3): {list(communities_louvain)[:3]}")
        except Exception as e:
            print("Error al detectar comunidades con Louvain:", e)
            communities = []

        ## Generación grafico

        plt.figure(figsize=(10, 7))
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_size=50)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        plt.title("Visualización básica del grafo de interacciones")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


        # Guardamos la datos

        self.top_out_degree = top_out
        self.top_in_degree = top_in
        self.top_betweenness = top5_node_centrality
        self.top_closeness = top5_closeness
        self.top_eigenvector = top5_eigevector

    def chat_local_llm(self, prompt: str = None):

        #Nombre del modelo
        model_name = "gemma-2-2b-it"
        #Obtenemos el token correspondiente al modelo
        token = AutoTokenizer.from_pretrained(model_name)
        #Descargamos el modelo
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Ponemos el modelo en modo evaluación para que no actualice los pesos
        model.eval();
        # Definimos que use GPU si tenemos disponibles
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        print("Modelo cargado. ¡Listo para chatear!\n")
        print("Escribe 'exit' o 'quit' para salir.\n")

        # Generamos un bucle para generar mensajes

        while True:
            prompt = input("Tú: ")

            if prompt.lower().strip() in ["exit", "quit"]:
                print("Saliendo del chat...")
                break

            # Genera el input de entra de forma que el modelo pueda entenderlo
            input_ids = token.encode(prompt, return_tensors="pt").to(device)

            #Desactivamos temporalmente el calculo de la gradiente
            with torch.no_grad():
                output_ids = model.generate(
                    input_ids,
                    max_length=128,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    pad_token_id=token.eos_token_id
                )

            generated_text = token.decode(output_ids[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()

            print(f"LLM: {response}\n")

    def generate_prompt_from_network(self, G: nx.DiGraph) -> str:

        if G.number_of_nodes() == 0:
            return "El grafo está vacío. No se puede generar un análisis."

        if not hasattr(self, "top_out_degree") or not hasattr(self, "analytics"):
            return "Faltan datos de análisis. No se ha ejecutado correctamente 'analyze_network' y 'analytics_hashtags_extended'."

        # obtenemos el top 3
        top3_users = self.top_out_degree[:3]

        usuarios = [f"{user} (menciones salientes: {grado})" for user, grado in top3_users]

        overall_df = self.analytics.get("overall")

        if overall_df is not None and not overall_df.empty:
            hashtag_texto = overall_df.iloc[0]["hashtag"]
        else:
            hashtag_texto = "ninguno"

        prompt = (
                "Se ha analizado una red de interacciones en Twitter basada en menciones.\n\n"
                f"Los 3 usuarios más activos (por menciones salientes) son:\n"
                + "\n".join(f" - {u}" for u in usuarios) +
                f"\n\nEl hashtag más frecuente en la red es: #{hashtag_texto}\n\n"
                "¿Qué factores podrían explicar por qué estos usuarios lideran las menciones "
                "y por qué este hashtag domina la conversación?\n"
                "Responde con un análisis interpretativo."
        )

        return prompt


















