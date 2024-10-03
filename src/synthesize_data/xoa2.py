import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from synthesizer import *
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test

# process adult data
x_original, y_original = load_adult()
target_name = y_original.columns[0]
y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']

# print(x_original.shape)
# print(x_original.isnull().sum())


# train test split
data = pd.concat([x_original, y_original], axis=1)
xtrain, xtest, ytrain, ytest = create_train_test.create_train_test(data, target_name=target_name, test_size=0.2, random_state=42)

