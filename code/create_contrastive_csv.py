import numpy as np
import pandas as pd
from load_dataset import give_dataframe



samples_per_number = 5000


df = give_dataframe()

data = np.zeros(shape=(samples_per_number*2*10, 3), dtype=np.object)

for n in range(10):
    same = (df.path[df.label==n]).sample(samples_per_number).values
    
    different = (df.path[df.label!=n]).sample(samples_per_number).values
    
    same_array = np.stack([same, np.random.permutation(same), np.ones(samples_per_number)], axis=1)
    
    diff_array = np.stack([same, different, np.zeros(samples_per_number)], axis=1)
    
    data[n*2*samples_per_number:(n+1)*2*samples_per_number] = np.vstack([same_array, diff_array])
    
data = pd.DataFrame(data, columns=['path1', 'path2', 'label'])

data['label'] = data.label.astype(np.int8)

data.to_csv('contrastive.csv', index=False)

