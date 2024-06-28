from synthesizer import *
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_adult, load_news, load_census



def main():
  # Load dataset
  x_original, y_original = load_news()
  target_name = y_original.columns[0]
  print(target_name)
  # One-hot encode
  categorical_columns = [] # there is no categorical columns in the news dataset
#   categorical_columns = ['workclass', 'education', 'marital-status', 'occupation', 
#                                         'relationship', 'race', 'sex', 'native-country', 'income']


  # Load synthesizer

  # Train new synthesizer and synthesize data

  synthesize_adult = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='BN_BE',
                             target_name=target_name, synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
                             csv_file_name='../data/news/news_BN_BE.csv', BN_filename='../data/news/news_BN_BE_model.pkl', verbose=True,
                             show_network=True)

  # synthesize_adult = synthesize_from_trained_model(x_original, y_original, categorical_columns,
  #                           sample_size=100_000, target_synthesizer='BN_BE',
  #                           target_name=' shares', synthesizer_file_name='../sdv trained model/news/news_synthesizer_onlyX.pkl',
  #                           csv_file_name='../data/news/news_BN_BE.csv', BN_model='../data/news/news_BN_BE_model.pkl', verbose=True)


if __name__ == '__main__':
  main()



