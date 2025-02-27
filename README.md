# 48935646

from sklearn.datasets import load_breast_cancer  
from sklearn.model_selection import train_test_split  
import pandas as pd  
import numpy as np  

SEED = 42  

cancer_data = load_breast_cancer()

df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)

df['target'] = cancer_data.target

train_df, temp_df = train_test_split(df, test_size=20, random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED)

print("Training set size:", train_df.shape)
print("Validation set size:", val_df.shape)
print("Test set size:", test_df.shape)
