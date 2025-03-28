# from synthesize_data.create_synthetic_data import adult_old
from synthesizer import *
import sys
import os
import pandas as pd
from create_synthetic_data import news, census, covertype, intrusion, credit, adult, mnist28, mnist12, census_kdd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from datasets import load_adult, load_news, load_census, load_covertype, load_intrusion

def single_run():
  # adult.CreateSyntheticDataAdult().create_synthetic_data()
  # census.CreateSyntheticDataCensus().create_synthetic_data()
  # credit.CreateSyntheticDataCredit().create_synthetic_data()
  # news.create_synthetic_data_news()
  # covertype.create_synthetic_data_covertype()
  # intrusion.create_synthetic_data_intrusion()
  # credit.create_synthetic_data_credit()
  # mnist28.CreateSyntheticDataMnist28().create_synthetic_data()
  # mnist12.CreateSyntheticDataMnist12().create_synthetic_data()
  # census_kdd.CreateSyntheticDataCensusKdd().create_synthetic_data()



  # adult.CreateSyntheticDataAdult().create_synthetic_data_pca_gmm()
  # census.CreateSyntheticDataCensus().create_synthetic_data_pca_gmm()
  # covertype.CreateSyntheticDataCovertype().create_synthetic_data_pca_gmm()
  # credit.CreateSyntheticDataCredit().create_synthetic_data_pca_gmm()
  # intrusion.CreateSyntheticDataIntrusion().create_synthetic_data_pca_gmm()
  # mnist12.CreateSyntheticDataMnist12().create_synthetic_data_pca_gmm()
  # mnist28.CreateSyntheticDataMnist28().create_synthetic_data_pca_gmm()
  # mnist28.CreateSyntheticDataMnist28().synthesize_categorical_pca_gmm_from_trained_model()
  # mnist12.CreateSyntheticDataMnist12().synthesize_categorical_pca_gmm_from_trained_model()
  # news.CreateSyntheticDataNews().create_synthetic_data_pca_gmm()

  # adult.CreateSyntheticDataAdult().create_synthetic_data_tvae_only()
  # census.CreateSyntheticDataCensus().create_synthetic_data_tvae_only()
  # census_kdd.CreateSyntheticDataCensusKdd().create_synthetic_data_tvae_only()
  # credit.CreateSyntheticDataCredit().create_synthetic_data_tvae_only()
  # covertype.CreateSyntheticDataCovertype().create_synthetic_data_tvae_only()
  # intrusion.CreateSyntheticDataIntrusion().create_synthetic_data_tvae_only()
  # mnist28.CreateSyntheticDataMnist28().create_synthetic_data_tvae_only()
  # mnist12.CreateSyntheticDataMnist12().create_synthetic_data_tvae_only()
  # news.CreateSyntheticDataNews().create_synthetic_data_tvae_only()

  # adult.CreateSyntheticDataAdult().create_synthetic_data_ensemble()
  # census.CreateSyntheticDataCensus().create_synthetic_data_ensemble()
  # census_kdd.CreateSyntheticDataCensusKdd().create_synthetic_data_ensemble()
  # credit.CreateSyntheticDataCredit().create_synthetic_data_ensemble()
  # covertype.CreateSyntheticDataCovertype().create_synthetic_data_ensemble() # done
  # intrusion.CreateSyntheticDataIntrusion().create_synthetic_data_ensemble()
  # mnist28.CreateSyntheticDataMnist28().create_synthetic_data_ensemble()
  # mnist12.CreateSyntheticDataMnist12().create_synthetic_data_ensemble()
  # news.CreateSyntheticDataNews().create_synthetic_data_ensemble()
  
  # adult.CreateSyntheticDataAdult().create_comparison_from_trained_model()
  # census.CreateSyntheticDataCensus().create_comparison_from_trained_model()
  # census_kdd.CreateSyntheticDataCensusKdd().create_comparison_from_trained_model()
  # credit.CreateSyntheticDataCredit().create_comparison_from_trained_model()
  # covertype.CreateSyntheticDataCovertype().create_comparison_from_trained_model()
  # intrusion.CreateSyntheticDataIntrusion().create_comparison_from_trained_model()
  # mnist28.CreateSyntheticDataMnist28().create_comparison_from_trained_model()
  # mnist12.CreateSyntheticDataMnist12().create_comparison_from_trained_model()
  # news.CreateSyntheticDataNews().create_comparison_from_trained_model()



  adult.CreateSyntheticDataAdult(feature_synthesizer='TVAE').create_synthetic_data()
  census.CreateSyntheticDataCensus(feature_synthesizer='TVAE').create_synthetic_data()
  census_kdd.CreateSyntheticDataCensusKdd(feature_synthesizer='TVAE').create_synthetic_data()
  credit.CreateSyntheticDataCredit(feature_synthesizer='TVAE').create_synthetic_data()
  covertype.CreateSyntheticDataCovertype(feature_synthesizer='TVAE').create_synthetic_data()
  # intrusion.CreateSyntheticDataIntrusion(feature_synthesizer='TVAE').create_synthetic_data()
  mnist28.CreateSyntheticDataMnist28(feature_synthesizer='TVAE').create_synthetic_data()
  mnist12.CreateSyntheticDataMnist12(feature_synthesizer='TVAE').create_synthetic_data()
  news.CreateSyntheticDataNews(feature_synthesizer='TVAE').create_synthetic_data()


  
  # pass




def main():
  # adult.create_synthetic_data_adult()
  # news.create_synthetic_data_news()
  # census.create_synthetic_data_census()
  # covertype.create_synthetic_data_covertype()
  # intrusion.create_synthetic_data_intrusion()
  # credit.create_synthetic_data_credit()
  pass


def create_synthetic_simulated():
  simulated_names = ['grid', 'gridr', 'ring']
  for name in simulated_names:

    data = pd.read_csv(f'../SDGym-research/data/simulated/{name}.csv')
    target_name = 'label'
    x_original = data.drop(columns=[target_name])
    y_original = data[target_name]
    categorical_columns = []

    synthesized_data = synthesize_data(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='gaussianNB',
                              target_name=target_name, synthesizer_file_name=f'../sdv trained model/simulated/{name}_synthesizer_onlyX.pkl',
                              csv_file_name=f'../SDGym-research/data/SDV_gaussian/{name}_SDV_gaussian_100k.csv', verbose=True,
                              npz_file_name=f'../SDGym-research/data/SDV_gaussian/{name}_300_300.npz')


    data = pd.read_csv(f'../SDGym-research/data/simulated/{name}.csv')
    target_name = 'label'
    x_original = data.drop(columns=[target_name])
    y_original = data[target_name]
    categorical_columns = []

    synthesized_data = synthesize_from_trained_model(x_original, y_original, categorical_columns,
                              sample_size=100_000, target_synthesizer='categoricalNB',
                              target_name=target_name, synthesizer_file_name=f'../sdv trained model/simulated/{name}_synthesizer_onlyX.pkl',
                              csv_file_name=f'../SDGym-research/data/SDV_categorical/{name}_SDV_categorical_100k.csv', verbose=True,
                              npz_file_name=f'../SDGym-research/data/SDV_categorical/{name}_300_300.npz')


if __name__ == '__main__':
  # main()
  print("RUN THIS SCRIPT FROM /SRC/")
  print("the command should be python synthesize_data/main.py")
  single_run()