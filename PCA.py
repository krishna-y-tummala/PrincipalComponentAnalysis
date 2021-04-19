#Change directory if necessary
import os
os.getcwd()
os.chdir('C:\\Users\\User\\Desktop\\school\\Python\\projects\\PCA')
os.getcwd()

#IMPORTS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.core.tools.datetimes import Scalar
import seaborn as sns

sns.set_style('whitegrid')

#LOAD DATA
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
cancer.keys()
print('\n',cancer['DESCR'])

df = pd.DataFrame(data=cancer['data'],columns=cancer['feature_names'])

print('\n',df.head(),'\n')

print('\n',df.info(),'\n')

print('\n',df.describe(),'\n')

#WE have 30 columns, most of which are not in scale

#We perform Principal Component Analysis to capture majority of the variance in the minimum number of components
#We need to scaled our data first

#SCALING
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)

scaled_df = pd.DataFrame(data=scaled_data,columns=cancer['feature_names'])

print('\n',scaled_df.head(),'\n')

#PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
#PCA(n_components=0.90,svd_solver='full')
#The better way to use PCA based on the amount of variance required is shown in the note above
#If 0 < n_components < 1 and svd_solver = 'full', the n_components input is the amount of variance required in the PCs.

pca.fit(scaled_df)

x_pca = pca.transform(scaled_df)

print('\n',x_pca.shape,'\n')

#We have 569 rows for each row in df and 2 columns for the PC weights

plt.figure(figsize=(10,6))
plt.scatter(x=x_pca[:,0],y=x_pca[:,1],c=cancer['target'],cmap='coolwarm')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.savefig('PCA.jpg')
plt.show()

#PCs
PC = pd.DataFrame(data=x_pca[:,0],columns=['First PC'])
PC['Second PC'] = x_pca[:,1]

print('\n',PC.head(),'\n')

#Component weights
df_components = pd.DataFrame(data=pca.components_,columns=cancer['feature_names'])
print('\n',df_components,'\n')

#HEATMAP of Variable Contribution
plt.figure(figsize=(10,6))
sns.heatmap(data=df_components,cmap='plasma')
plt.savefig('Heatmap.jpg')
plt.show()

print('\nO and 1 denote the 1st and 2nd Principal Components for each variable in the scaled data\n')
print('\nThe lighter colors have the most weights and the darker colors have the least weights on the Principal Components\n')
print('\nNow the Principal Components can be passed into any Supervised algorithm to reduce overfitting')

