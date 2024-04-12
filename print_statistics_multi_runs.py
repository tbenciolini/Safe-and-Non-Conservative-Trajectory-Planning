import pickle
import numpy as np

np.set_printoptions(suppress=True)
file_names = [
                'USA_US101-13_2_T-1_20-10-2023_09:36:48_variables_many_runs',
                'DEU_A99-1_2_T-1_20-10-2023_09:35:33_variables_many_runs',
            ]
for file_name in file_names:
    outfile = open('data/stats_'+file_name+'.txt','a')
    print(file_name,file=outfile)

    store_variables = pickle.load(open('data/'+file_name, 'rb'))
    for key in store_variables.keys():
        values_runs = store_variables[key]
        unit_str = ''
        if key[2:] not in ['cost', 'cost_comparison']:
            values_runs = [1000*val for val in values_runs] #   plot in ms
            unit_str = ' [ms]'
        print(key[2:]+unit_str,'mean', '%.3f'%np.average(values_runs),
                '\pm', '%.3f'%np.std(values_runs),file=outfile)