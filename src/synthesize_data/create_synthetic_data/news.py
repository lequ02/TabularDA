import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
import pandas as pd


def create_synthetic_data_news():
  # synthesize data for the news dataset 
  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  x_original = pd.concat([x_original, y_original], axis=1)
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer.pkl',
                            csv_file_name='../data/news/news_sdv_100k.csv', verbose=True,
                            show_network=True)

  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv_gaussian_100k = synthesize_data(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='gaussianNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
                            csv_file_name='../data/news/news_sdv_gaussian_100k.csv', verbose=True,
                            show_network=True)

  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv_categorical_100k = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                            sample_size=100_000, target_synthesizer='categoricalNB',
                            target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
                            csv_file_name='../data/news/news_sdv_categorical_100k.csv', verbose=True,
                            show_network=True)

  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  categorical_columns = [] # there is no categorical columns in the news dataset
  synthesize_news_sdv_BN_MLE = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='BN_MLE',
                             target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
                             csv_file_name='../data/news/news_BN_BE.csv', BN_filename='../data/news/news_BN_MLE_model.pkl', verbose=True,
                             show_network=True)