import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing

dataframe = pd.read_csv(r"C:\Users\David\Documents\coding\Project\Tubes-AAMI\data_dummy2.csv")
dataframe['Timestamp'] = pd.to_datetime(dataframe['Timestamp'])
dataframe.set_index('Timestamp', inplace=True)
scaler = preprocessing.MinMaxScaler()

data_rescaled = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns, index=dataframe.index)
pca = PCA().fit(data_rescaled)


import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, 4, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Number of Components')
plt.xticks(np.arange(0, 11, step=1)) #change from 0-based array index to 1-based human-readable label
plt.ylabel('Cumulative variance (%)')
plt.title('The number of components needed to explain variance')

plt.axhline(y=0.9, color='r', linestyle='-')
plt.text(0.5, 0.99, '90% cut-off threshold', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()