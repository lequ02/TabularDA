import sys
import os
import pandas as pd
from create_synthetic_data import CreateSyntheticData
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype, load_credit

class CreateSyntheticDataNews(CreateSyntheticData.CreateSyntheticData):
    def __init__(self, feature_synthesizer = 'CTGAN'):
        ds_name = 'news'
        # initially synthesize the data with no categorical columns
        categorical_columns = []

        numerical_columns_pca_gmm = [' LDA_00', ' LDA_01', ' LDA_02', ' LDA_03', ' LDA_04',
       ' abs_title_sentiment_polarity', ' abs_title_subjectivity',
       ' average_token_length', ' avg_negative_polarity',
       ' avg_positive_polarity', ' global_rate_negative_words',
       ' global_rate_positive_words', ' global_sentiment_polarity',
       ' global_subjectivity', ' kw_avg_avg', ' kw_avg_max', ' kw_avg_min',
       ' kw_max_avg', ' kw_max_max', ' kw_max_min', ' kw_min_avg',
       ' kw_min_max', ' kw_min_min', ' max_negative_polarity',
       ' max_positive_polarity', ' min_negative_polarity',
       ' min_positive_polarity', ' n_non_stop_unique_tokens',
       ' n_non_stop_words', ' n_tokens_content', ' n_tokens_title',
       ' n_unique_tokens', ' num_hrefs', ' num_imgs', ' num_keywords',
       ' num_self_hrefs', ' num_videos', ' rate_negative_words',
       ' rate_positive_words', ' self_reference_avg_sharess',
       ' self_reference_max_shares', ' self_reference_min_shares',
       ' title_sentiment_polarity', ' title_subjectivity']
        
        super().__init__(ds_name, load_news, ' shares', categorical_columns=categorical_columns, features_synthesizer=feature_synthesizer,
                            numerical_cols_pca_gmm=numerical_columns_pca_gmm,
                            sample_size_to_synthesize=100_000, missing_values_strategy='drop', test_size=0.2, is_classification=False)
        
    # def create_synthetic_data_pca_gmm(self):
    #     xtrain, xtest, ytrain, ytest, target_name, categorical_columns = self.read_data()
    #     self.synthesize_from_trained_model(xtrain, ytrain, categorical_columns, 'sdv_pca_gmm', 'pca_gmm', is_classification=False)
        