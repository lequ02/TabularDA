from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork
import numpy as np
import pandas as pd
import time



values = pd.DataFrame(np.random.randint(low=0, high=11, size=(100_000, 6)),
                      columns=['A', 'B', 'C', 'D', 'E', 'F'])

values['T'] = np.random.randint(low=0, high=1000, size=(100_000))
model = BayesianNetwork([('A', 'B'), ('A', 'T'), 
                        ('B', 'D'), ('B', 'T'),
                        ('C', 'E'), ('C', 'T'),
                        ('D', 'T'),
                        ('E', 'T'),
                        ('F', 'T')])
model.fit(values)
inference = VariableElimination(model)
print(model.nodes())

start_time = time.time()
for i, row in values.iterrows():
    if i==1:
        break
    evidence = {var: val for var, val in row.to_dict().items() if var in model.nodes() and var != 'T'}

    phi_query = inference.map_query(variables=['T'], evidence = evidence )
    print(phi_query)
# Code to be timed
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time, "seconds")

# for i in range(1000):
#     evidence = {'A': values['A'][i], 'B': values['B'][i], 'C': values['C'][i], 'D': values['D'][i], 'F': values['F'][i], 'E': values['E'][i]}
#     phi_query = inference.map_query(variables=['T'], evidence = {'A':1, 'B':4, 'C':0, 'D': 1, 'F':0, 'E':3} )


print(type(phi_query))
print(len(phi_query))
print(phi_query)