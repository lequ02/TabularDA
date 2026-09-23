from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def create_path_dict(dataset_name, target_name):
    root = PROJECT_ROOT / 'data' / 'corrected_v2' / dataset_name / 'seed_{seed}'
    return {
        'train_original': f'{root}/onehot_{dataset_name}_train.csv',
        'dev': f'{root}/onehot_{dataset_name}_dev.csv',
        'test': f'{root}/onehot_{dataset_name}_test.csv',
        'split_manifest': f'{root}/split_manifest.json',
        'synthetic': {
            'ctgan': f'{root}/onehot_{dataset_name}_sdv_100k.csv',
            'categorical': f'{root}/onehot_{dataset_name}_sdv_categorical_100k.csv',
            'gaussian': f'{root}/onehot_{dataset_name}_sdv_gaussian_100k.csv',
            'pca_gmm': f'{root}/onehot_{dataset_name}_sdv_pca_gmm_100k.csv',
            'xgb': f'{root}/onehot_{dataset_name}_sdv_xgb_100k.csv',
            'rf': f'{root}/onehot_{dataset_name}_sdv_rf_100k.csv',
            'tvae': f'{root}/onehot_{dataset_name}_sdv_tvae_100k.csv',
            'tvae_gaussian': f'{root}/onehot_{dataset_name}_sdv_tvae_gaussian_100k.csv',
            'tvae_categorical': f'{root}/onehot_{dataset_name}_sdv_tvae_categorical_100k.csv',
            'tvae_pca_gmm': f'{root}/onehot_{dataset_name}_sdv_tvae_pca_gmm_100k.csv',
            'tvae_xgb': f'{root}/onehot_{dataset_name}_sdv_tvae_xgb_100k.csv',
            'tvae_rf': f'{root}/onehot_{dataset_name}_sdv_tvae_rf_100k.csv',
            'compare_categorical': f'{root}/onehot_{dataset_name}_sdv_compare_categoricalNB_100k.csv',
            'compare_gaussian': f'{root}/onehot_{dataset_name}_sdv_compare_gaussianNB_100k.csv',
            'compare_pca_gmm': f'{root}/onehot_{dataset_name}_sdv_compare_pca_gmm_100k.csv',
            'compare_xgb': f'{root}/onehot_{dataset_name}_sdv_compare_xgb_100k.csv',
            'compare_rf': f'{root}/onehot_{dataset_name}_sdv_compare_rf_100k.csv',
        },
        'target_name': target_name,
    }


IN_DATA_PATHS = {
    'adult': create_path_dict('adult', 'income'),
    'census': create_path_dict('census', 'income'),
    'census_kdd': create_path_dict('census_kdd', 'income'),
    'news': create_path_dict('news', ' shares'),
    'covertype': create_path_dict('covertype', 'Cover_Type'),
    'intrusion': create_path_dict('intrusion', 'target'),
    'credit': create_path_dict('credit', 'Class'),
    'mnist12': create_path_dict('mnist12', 'label'),
    'mnist28': create_path_dict('mnist28', 'label'),
}

# Preserve the established MNIST method names while reading only corrected_v2 artifacts.
for _dataset in ('mnist12', 'mnist28'):
    _root = PROJECT_ROOT / 'data' / 'corrected_v2' / _dataset / 'seed_{seed}'
    IN_DATA_PATHS[_dataset]['synthetic'].update({
        'pca_gmm_num': f'{_root}/onehot_{_dataset}_sdv_pca_gmm_num_100k.csv',
        'pca_gmm_cat': f'{_root}/onehot_{_dataset}_sdv_pca_gmm_cat_100k.csv',
    })

OUT_DATA_PATHS = "../../output/"
