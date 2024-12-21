IN_DATA_PATHS = {
    'adult': {
        'train_original': '../../data/adult/onehot_adult_train.csv',
        'test': '../../data/adult/onehot_adult_test.csv',
        'synthetic': {
            'ctgan': '../../data/adult/onehot_adult_sdv_100k.csv',
            'categorical': '../../data/adult/onehot_adult_sdv_categorical_100k.csv',
            'gaussian': '../../data/adult/onehot_adult_sdv_gaussian_100k.csv'
        },
        'target_name': 'income'
    },
    'census': {
        'train_original': '../../data/census/onehot_census_train.csv',
        'test': '../../data/census/onehot_census_test.csv',
        'synthetic': {
            'ctgan': '../../data/census/onehot_census_sdv_100k.csv',
            'categorical': '../../data/census/onehot_census_sdv_categorical_100k.csv',
            'gaussian': '../../data/census/onehot_census_sdv_gaussian_100k.csv'
        },
        'target_name': 'income'
    },
    'news': {
        'train_original': '../../data/news/onehot_news_train.csv',
        'test': '../../data/news/onehot_news_test.csv',
        'synthetic': {
            'ctgan': '../../data/news/onehot_news_sdv_100k.csv',
            'categorical': '../../data/news/onehot_news_sdv_categorical_100k.csv',
            'gaussian': '../../data/news/onehot_news_sdv_gaussian_100k.csv'
        },
        'target_name': ' shares'
    },
    'mnist12': {
        'train_original': '../../data/mnist12/onehot_mnist12_train.csv',
        'test': '../../data/mnist12/onehot_mnist12_test.csv',
        'synthetic': {
            'ctgan': '../../data/mnist12/onehot_mnist12_sdv_100k.csv',
            'categorical': '../../data/mnist12/onehot_mnist12_sdv_categorical_100k.csv',
            'gaussian': '../../data/mnist12/onehot_mnist12_sdv_gaussian_100k.csv'
        },
        'target_name': 'label'
    },
    'mnist28': {
        'train_original': '../../data/mnist28/onehot_mnist28_train.csv',
        'test': '../../data/mnist28/onehot_mnist28_test.csv',
        'synthetic': {
            'ctgan': '../../data/mnist28/onehot_mnist28_sdv_100k.csv',
            'categorical': '../../data/mnist28/onehot_mnist28_sdv_categorical_100k.csv',
            'gaussian': '../../data/mnist28/onehot_mnist28_sdv_gaussian_100k.csv'
        },
        'target_name': 'label'
    },
}

OUT_DATA_PATHS = "../../output/"