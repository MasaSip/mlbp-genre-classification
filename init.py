# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data_df = pd.read_csv('train_data.csv',header=None)
data = np.asarray(data_df)
labels_df = pd.read_csv('train_labels.csv',header=None)
labels = np.asarray(labels_df)
