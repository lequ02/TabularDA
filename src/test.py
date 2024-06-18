from data_loader import data_loader
from datasets import load_news, load_adult, load_census
import pandas as pd
from synthesize_data.onehot import onehot

# numerical columns to normalize
numerical_columns = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

train = data_loader('census', 'sdv_categorical', 100, numerical_columns = numerical_columns).train_data
test = data_loader('census', 'sdv_categorical', 100, numerical_columns = numerical_columns).test_data

print(train)
print(test)

# Print the first batch of the training data
for i, (inputs, labels) in enumerate(train):
    print(f"Batch {i+1}:")
    print("Inputs:", inputs)
    print("Labels:", labels)
    break  # We break the loop after the first batch


# x , y = load_census()
# print(x.columns.unique())
# print(x.workclass.unique())
# y['income'] = y['income'].map({'<=50K': 0, '>50K': 1})


# x_onehot, _  = onehot(x, x.copy(), ['workclass', 'education', 'marital-status', 'occupation',
#                         'relationship', 'race', 'sex', 'native-country'], verbose=True)

# x_onehot = x_onehot.reindex(sorted(x_onehot.columns), axis=1)
# print(y.income.unique())

# x = pd.concat([x_onehot, y], axis=1)
# print(x.head())





# # Assuming df is your DataFrame
# links = ['../data./adults_ori_onehot.csv',
#         '../data/june17-adult_sdv_100k_onlyX.csv',
#         '../data/onehot_adult_sdv_categorical_100k.csv',
#         '../data/onehot_adult_sdv_gaussian_100k.csv',
#          '../data/onehot_census_sdv_categorical_100k.csv',
#          '../data/onehot_census_sdv_gaussian_100k.csv',]

# for link in links:
#     df = pd.read_csv(link, index_col=0)

#     # Remove the repetition in all the column names
#     df.columns = [col.split('_', 1)[-1] if col.count('_') > 1 else col for col in df.columns]

#     # Print the updated column names
#     print("\n", link)
#     print(df.columns)
#     df.to_csv(link)







# adult, _ = load_adult()
# census, _ = load_census()   

# # print(adult.columns.unique())
# # print(adult.workclass.unique())
# # print(adult.workclass[0])

# link = '../data/adult/old/adult_sdv_100k.csv'
# df = pd.read_csv('../data/adult/old/adult_sdv_100k.csv', index_col=0)
# df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
# df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})

# # print(df.head())


# _, dfx = onehot(adult, df.iloc[:, :-1], ['workclass', 'education', 'marital-status', 'occupation', 
#                                         'relationship', 'race', 'sex', 'native-country'], verbose=True)


# dfx = dfx.reindex(sorted(dfx.columns), axis=1)
# df = pd.concat([dfx, df['income']], axis=1)

# print(dfx.columns.unique())
# print(df.income.unique())
# print(df.head())
# df.to_csv('../data/adult/old/onehot_adult_sdv_100k.csv')
# print("\n")


# link1 = '../data/census/old/census_sdv_100k.csv'
# df1 = pd.read_csv('../data/census/old/census_sdv_100k.csv', index_col=0)
# df1['income'] = df1['income'].str.replace('.', '', regex=False)
# df1['income'] = df1['income'].map({'<=50K': 0, '>50K': 1})

# _, df1x = onehot(census, df1.iloc[:,:-1], ['workclass', 'education', 'marital-status', 'occupation',
#                                             'relationship', 'race', 'sex', 'native-country'], verbose=True)

# df1x = df1x.reindex(sorted(df1x.columns), axis=1)
# df1 = pd.concat([df1x, df1['income']], axis=1)

# print(df1x.columns.unique())
# print(df1.income.unique())
# print(df1.head())
# df1.to_csv('../data/census/old/onehot_census_sdv_100k.csv')
