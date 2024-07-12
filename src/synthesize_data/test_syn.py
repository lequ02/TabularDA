from synthesizer import *
import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import load_adult, load_news, load_census



def main():

# D:\SummerResearch\SDGym-research\data\simulated\grid.csv


  simulated_names = ['grid', 'gridr', 'ring']
  for name in simulated_names:

    data = pd.read_csv(f'../SDGym-research/data/simulated/{name}.csv')
    target_name = 'label'
    x_original = data.drop(columns=[target_name])
    y_original = data[target_name]
    categorical_columns = []

    synthesized_data = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='gaussianNB',
                              target_name=target_name, synthesizer_file_name=f'../sdv trained model/simulated/{name}_synthesizer_onlyX.pkl',
                              csv_file_name=f'../SDGym-research/data/SDV_gaussian/{name}_SDV_gaussian_100k.csv', verbose=False,
                              npz_file_name=f'../SDGym-research/data/SDV_gaussian/{name}_300_300.npz')

    print("\ngaussian unique labels:", synthesized_data.label.unique())
    print("0:", sum(synthesized_data.label==0))
    print("1:", sum(synthesized_data.label==1))


    data = pd.read_csv(f'../SDGym-research/data/simulated/{name}.csv')
    target_name = 'label'
    x_original = data.drop(columns=[target_name])
    y_original = data[target_name]
    categorical_columns = []

    synthesized_data = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='categoricalNB',
                              target_name=target_name, synthesizer_file_name=f'../sdv trained model/simulated/{name}_synthesizer_onlyX.pkl',
                              csv_file_name=f'../SDGym-research/data/SDV_categorical/{name}_SDV_categorical_100k.csv', verbose=False,
                              npz_file_name=f'../SDGym-research/data/SDV_categorical/{name}_300_300.npz')


    print("\ncategorical unique labels:", synthesized_data.label.unique())
    print("0:", sum(synthesized_data.label==0))
    print("1:", sum(synthesized_data.label==1))


if __name__ == '__main__':
  main()



