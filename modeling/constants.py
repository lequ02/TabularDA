IN_DATA_PATHS = {
    'adult': {
        'train_original': '../../data/adult/adult_train.csv',
        'test': '../../data/adult/adult_test.csv',
        'synthetic': {
            'ctgan': '../../data/adult/onehot_adult_sdv_100k.csv',
            'categorical': '../../data/adult/onehot_adult_sdv_categorical_100k.csv',
            'gaussian': '../../data/adult/onehot_adult_sdv_gaussian_100k.csv'
        }
    },
    'census': {
        'train_original': '../../data/census/census_train.csv',
        'test': '../../data/census/census_test.csv',
        'synthetic': {
            'ctgan': '../../data/census/onehot_census_sdv_100k.csv'
        }
    },
    'news': {
        'train_original': '../../data/news/news_train.csv',
        'test': '../../data/news/news_test.csv',
        'synthetic': {
            'ctgan': '../../data/news/onehot_news_sdv_100k.csv',
            'categorical': '../../data/news/onehot_news_sdv_categorical_100k.csv',
            'gaussian': '../../data/news/onehot_news_sdv_gaussian_100k.csv'
        }
    }
}

OUT_DATA_PATHS = "../../output/"