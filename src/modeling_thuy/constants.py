def create_path_dict(dataset_name, target_name):
    return {
        'train_original': f'../../data/{dataset_name}/onehot_{dataset_name}_train.csv',
        'test': f'../../data/{dataset_name}/onehot_{dataset_name}_test.csv',
        'synthetic': {
            'ctgan': f'../../data/{dataset_name}/onehot_{dataset_name}_sdv_100k.csv',
            'categorical': f'../../data/{dataset_name}/onehot_{dataset_name}_sdv_categorical_100k.csv',
            'gaussian': f'../../data/{dataset_name}/onehot_{dataset_name}_sdv_gaussian_100k.csv',
            'pca_gmm': f'../../data/{dataset_name}/onehot_{dataset_name}_sdv_pca_gmm_100k.csv',
            'xgb': f'../../data/{dataset_name}/onehot_{dataset_name}_sdv_xgb_100k.csv',
            'rf': f'../../data/{dataset_name}/onehot_{dataset_name}_sdv_rf_100k.csv',
            'tvae': f'../../data/{dataset_name}/onehot_{dataset_name}_sdv_tvae_100k.csv',

        },
        'target_name': target_name
    }


mnist12_paths = create_path_dict('mnist12', 'label')


IN_DATA_PATHS = {

    'adult' : create_path_dict('adult', 'income'),
    'census' : create_path_dict('census', 'income'),
    'census_kdd' : create_path_dict('census', 'income'),
    'news' : create_path_dict('news', ' shares'),
    # 'mnist12' : create_path_dict('mnist12', 'label'),
    # 'mnist28' : create_path_dict('mnist28', 'label'),
    'covertype': create_path_dict('covertype', 'Cover_Type'),
    'intrusion': create_path_dict('intrusion', 'target'),
    'credit': create_path_dict('credit', 'Class'),


    'mnist12': {
        'train_original': '../../data/mnist12/onehot_mnist12_train.csv',
        'test': '../../data/mnist12/onehot_mnist12_test.csv',
        'synthetic': {
            'ctgan': '../../data/mnist12/onehot_mnist12_sdv_100k.csv',
            'categorical': '../../data/mnist12/onehot_mnist12_sdv_categorical_100k.csv',
            'gaussian': '../../data/mnist12/onehot_mnist12_sdv_gaussian_100k.csv',
            'pca_gmm_num': '../../data/mnist12/onehot_mnist12_sdv_pca_gmm_num_100k.csv',
            'pca_gmm_cat': '../../data/mnist12/onehot_mnist12_sdv_pca_gmm_cat_100k.csv',
            'tvae': '../../data/mnist12/onehot_mnist12_sdv_tvae_only_100k.csv',
            'xgb': '../../data/mnist12/onehot_mnist12_sdv_xgb_100k.csv',
            'rf': '../../data/mnist12/onehot_mnist12_sdv_rf_100k.csv',
        },
        'target_name': 'label'
    },
    'mnist28': {
        'train_original': '../../data/mnist28/onehot_mnist28_train.csv',
        'test': '../../data/mnist28/onehot_mnist28_test.csv',
        'synthetic': {
            'ctgan': '../../data/mnist28/onehot_mnist28_sdv_100k.csv',
            'categorical': '../../data/mnist28/onehot_mnist28_sdv_categorical_100k.csv',
            'gaussian': '../../data/mnist28/onehot_mnist28_sdv_gaussian_100k.csv',
            'pca_gmm_num': '../../data/mnist28/onehot_mnist28_sdv_pca_gmm_num_100k.csv',
            'pca_gmm_cat': '../../data/mnist28/onehot_mnist28_sdv_pca_gmm_cat_100k.csv',
            'tvae': '../../data/mnist28/onehot_mnist28_sdv_tvae_only_100k.csv',
            'xgb': '../../data/mnist28/onehot_mnist28_sdv_xgb_100k.csv',
            'rf': '../../data/mnist28/onehot_mnist28_sdv_rf_100k.csv',
        },
        'target_name': 'label'
    },

}

OUT_DATA_PATHS = "../../output/"