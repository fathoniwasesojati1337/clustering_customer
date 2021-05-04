# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv('Mall_Customers.csv')
 
# tampilkan 3 baris pertama
df.head(3)

df = df.rename(columns={
    'Gender': 'jenis_kelamin',
    'Age': 'umur',
    'Annual Income (k$)' : 'pendapatan_income',
    'Spending Score (1-100)' : 'score_habis'
})

df['jenis_kelamin'].replace(['Female', 'Male'], [0,1], inplace=True)

X = df.drop(['CustomerID', 'jenis_kelamin'], axis=1)

# mencari cluster yang paling optimal

from sklearn.cluster import KMeans



import seaborn as sns

wcss = []

for i in range(1,15):
  k = KMeans(i)
  k.fit(X)
  wcss.append(k.inertia_)

# membuat plot inertia
sns.lineplot(x=list(range(1, 15)), y=wcss)

# cluseter siku atau eibow method yaitu 5

km = KMeans(n_clusters=5, random_state=42).fit(X)
y_means = km.predict(X)
y_means

sns.scatterplot(X['pendapatan_income'], X['score_habis'], hue= y_means, palette=sns.color_palette('hls', 5))
