# from pgmpy.inference import VariableElimination
# from pgmpy.models import BayesianNetwork
# import numpy as np
# import pandas as pd
# import time



# values = pd.DataFrame(np.random.randint(low=0, high=11, size=(100_000, 6)),
#                       columns=['A', 'B', 'C', 'D', 'E', 'F'])

# values['T'] = np.random.randint(low=0, high=1000, size=(100_000))
# model = BayesianNetwork([('A', 'B'), ('A', 'T'), 
#                         ('B', 'D'), ('B', 'T'),
#                         ('C', 'E'), ('C', 'T'),
#                         ('D', 'T'),
#                         ('E', 'T'),
#                         ('F', 'T')])
# model.fit(values)
# inference = VariableElimination(model)
# print(model.nodes())

# start_time = time.time()
# for i, row in values.iterrows():
#     if i==1:
#         break
#     evidence = {var: val for var, val in row.to_dict().items() if var in model.nodes() and var != 'T'}

#     phi_query = inference.map_query(variables=['T'], evidence = evidence )
#     print(phi_query)
# # Code to be timed
# end_time = time.time()
# execution_time = end_time - start_time
# print("Execution time:", execution_time, "seconds")

# # for i in range(1000):
# #     evidence = {'A': values['A'][i], 'B': values['B'][i], 'C': values['C'][i], 'D': values['D'][i], 'F': values['F'][i], 'E': values['E'][i]}
# #     phi_query = inference.map_query(variables=['T'], evidence = {'A':1, 'B':4, 'C':0, 'D': 1, 'F':0, 'E':3} )


# print(type(phi_query))
# print(len(phi_query))
# print(phi_query)


import pandas as pd

df = pd.read_csv('D:\\SummerResearch\\data\\adult\\onehot_adult_sdv_100k.csv')



x_null = df.isnull().sum()
print("\n\nNull values in xtrain: ", x_null[x_null > 0])


import sys
import os
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datasets import load_adult, load_news, load_census, load_covertype
from commons import create_train_test


x_original, y_original = load_adult()
target_name = y_original.columns[0]
y_original = y_original['income'].map({'<=50K': 0, '>50K': 1})
categorical_columns = ['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'sex', 'native-country']


null = x_original.isnull().sum()
print("\n\nNull values in xtrain: ", null[null > 0])