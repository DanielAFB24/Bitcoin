from DataExtractor import DataExtractor

# Paso 1: Crear extractor con un parámetro de búsqueda
extractor = DataExtractor()
extractor.load_data_api(query="Tesla", max_results=200)

# Paso 2: Analizar hashtags
resultados = extractor.analytics_hashtags_extended()

# Paso 3: Nube de palabras
extractor.generate_hashtag_wordcloud(resultados.get('overall'))

# Paso 4: Modelado de tópicos
extractor.model_topics(num_topics=5)

# Paso 5: Análisis de sentimiento
extractor.analyze_sentiment(method='textblob')

# Paso 6: Gráfico de distribución de sentimiento
extractor.plot_sentiment_distribution()

# Paso 7: Resumen extractivo
extractor.parse_and_summarize(summary_ratio=0.3)

# Paso 8: Árbol sintáctico del primer texto limpio
extractor.visualize_dependency_tree(index=0)

# Paso 9 Generar Grafo
G = extractor.build_interaction_graph()

# Paso 10 : Analisis Grafo
extractor.analyze_network(G)

# Paso 11 : Chat local
extractor.chat_local_llm()

# Paso 12 : Generar Pront
extractor.generate_prompt_from_network(G)



