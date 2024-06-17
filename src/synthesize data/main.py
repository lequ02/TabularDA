from synthesizer import synthesize_data, synthesize_from_trained_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_adult, load_news, load_census



def main():
  # Load dataset
  x_original, y_original = load_news()

  # One-hot encode
  categorical_columns = [] # there is no categorical columns in the news dataset

  # Load synthesizer

  # Train new synthesizer and synthesize data
  # synthesize_news = synthesize_data(x_original, y_original, categorical_columns,
  #                             sample_size=100_000, target_synthesizer='gaussianNB',
  #                            target_name='shares', synthesizer_file_name='news_synthesizer_onlyX.pkl',
  #                            csv_file_name='news_sdv_gaussion_100k.csv')

  # Synthesize data using trained synthesizer
  synthesize_news = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='categoricalNB',
                             target_name='shares', synthesizer_file_name='../sdv trained model/news_synthesizer_onlyX.pkl',
                             csv_file_name='../data/news/news_sdv_categorical_100k.csv')


if __name__ == '__main__':
  main()



