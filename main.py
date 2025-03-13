from DataExtractor import DataExtractor

extractor = DataExtractor("Bitcoin_tweets_dataset_2.csv", chunksize=100000)

# Cargar los datos del archivo CSV
extractor.load_data()
extractor.generate_hashtag_wordcloud()