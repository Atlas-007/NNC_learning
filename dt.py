import pandas as pd
import numpy as np

def save_iris_data(num_rows=5):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    iris = pd.read_csv(url, header=None, names=column_names)
    
    # Extract the first 'num_rows' rows and the first 4 columns
    X = iris.iloc[:num_rows, 0:4].values  
    
    np.save('iris_features.npy', X)  # Save the data to a .npy file

save_iris_data(num_rows=5)



