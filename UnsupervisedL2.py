
"""
Unsupervised Learning ..
-- DBSCAN clustering → detect outliers.
-- PCA (Dimensionality Reduction) → visualize high-dim data.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot  as plt 
from sklearn.decomposition import PCA 
from matplotlib.pyplot import subplot ,subplots_adjust 
from sklearn.cluster import KMeans 
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_circles  
from sklearn.metrics import silhouette_score 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import make_classification
from sklearn.metrics import mean_squared_error


"""
np.random.seed(42)

cluster_1 = np.random.randn(30, 2) + [2, 2]
cluster_2 = np.random.randn(30, 2) + [7, 7]
outliers = np.random.uniform(low=-2, high=12, size=(5, 2))

data = np.vstack([cluster_1, cluster_2, outliers])
df = pd.DataFrame(data, columns=['x', 'y'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['x', 'y']])

# DBSCAN

dbscan = DBSCAN(eps=0.5, min_samples=5)
df['cluster'] = dbscan.fit_predict(X_scaled)


print(df['cluster'].value_counts())

plt.figure(figsize=(6,6))
plt.scatter(df['x'], df['y'], c=df['cluster'], cmap='rainbow', s=50)
plt.title("DBSCAN Clustering (Outliers = -1)")
plt.xlabel('x')
plt.ylabel('y')
plt.show()
print('------------Seperate-----------')
#1
np.random.seed(42)

A = np.random.randn(40,2) * 0.5 + [2,2]
B = np.random.randn(40,2) * 0.6 + [7,7]
out = np.random.uniform(low=-2, high=11, size=(8,2))
data = np.vstack([A,B,out])
df = pd.DataFrame(data, columns=['x','y'])
df.head()
print(df)
print(df.shape)

x=df[['x','y']].values
scale=StandardScaler()
x_scaled=scale.fit_transform(x)

dbscan=DBSCAN(eps=0.5,min_samples=8)
df['cluster']=dbscan.fit_predict(x_scaled)

plt.scatter(df['x'],df['y'],c=df['cluster'],cmap='rainbow')
plt.show()
print(df['cluster'].sum())
print(df['cluster'].value_counts())
print('------------Seperate-----------')

#2

np.random.seed(0)
# tight cluster
C1 = np.random.randn(60,2) * 0.2 + [2,2]
# loose cluster
C2 = np.random.randn(60,2) * 1.2 + [8,2]
# medium cluster
C3 = np.random.randn(60,2) * 0.6 + [5,8]
data = np.vstack([C1,C2,C3])
df = pd.DataFrame(data, columns=['x','y'])
df.head()

print(df.shape)

scale=StandardScaler()
x_scaled=scale.fit_transform(df[['x', 'y']])

#dbscan=DBSCAN(eps=0.6,min_samples=6) # silhoutte = 0.71 
dbscan=DBSCAN(eps=0.5,min_samples=5) # silhoutte = 0.73 Clear 
df['cluster']=dbscan.fit_predict(x_scaled)

plt.scatter(df['x'],df['y'],c=df['cluster'],cmap='rainbow')
plt.show()

print(df['cluster'].value_counts())

print('Silhoutte Score => ',silhouette_score(x_scaled,df['cluster']))

print('------------Seperate-----------')

#3

X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05, random_state=1)
df = pd.DataFrame(X, columns=['x','y'])
df.head()

x=df[['x','y']].values

scale=StandardScaler()
x_scaled=scale.fit_transform(x)
print(df.shape)
dbscan=DBSCAN(eps=0.5,min_samples=20) 
df['cluster']=dbscan.fit_predict(x_scaled)

plt.scatter(df['x'],df['y'],c=df['cluster'],cmap='rainbow')
plt.show()

print('Silhoutte Score => ',silhouette_score(x_scaled,df['cluster']))
print(df['cluster'].value_counts())

#-- 
#With K-Mean
inertia=[]
K=range(1,11)
for k in K:
   kmean=KMeans(n_clusters=k,random_state=42)
   kmean.fit(x_scaled)
   inertia.append(kmean.inertia_)


plt.scatter(K,inertia)
plt.show()

kmeanean=KMeans(n_clusters=5,random_state=42)
kmean.fit(x_scaled)

df['cluster_kmean']=kmean.labels_
print('Silhoutte Score => ',silhouette_score(x_scaled,kmean.labels_))
print(df['cluster_kmean'].value_counts()) 
plt.scatter(x_scaled[:,0],x_scaled[:,1],c=df['cluster_kmean'])
plt.plot(x_scaled,kmean.predict(x_scaled) )
plt.show()

print('------------Seperate-----------')

#4
np.random.seed(7)
x1 = np.random.randn(100) * 0.5 + 5          # small scale
x2 = (np.random.randn(100) * 200) + 3000     # large scale
data = np.column_stack([x1, x2])
df = pd.DataFrame(data, columns=['feature_small','feature_large'])
df.head()
print(df.shape)
x=df[['feature_small','feature_large']].values

#without scalling 

dbscan=DBSCAN(eps=0.5,min_samples=2)
df['cluster']=dbscan.fit_predict(x)

plt.scatter(df['feature_small'],df['feature_large'],c=df['cluster'],cmap='rainbow')
plt.show() 

print("silhoutte Score without scalling => ",silhouette_score(x,df['cluster']))
print(df['cluster'].value_counts())

#with Scalling 
scale=StandardScaler()
x_scaled=scale.fit_transform(x)

dbscan=DBSCAN(eps=0.5,min_samples=8)
df['cluster']=dbscan.fit_predict(x_scaled)

plt.scatter(df['feature_small'],df['feature_large'],c=df['cluster'],cmap='rainbow')
plt.show()

print("silhoutte Score with scalling => ",silhouette_score(x_scaled,df['cluster']))
print(df['cluster'].value_counts())


#5
np.random.seed(11)
C1 = np.random.randn(80,3) + [2,2,2]
C2 = np.random.randn(80,3) + [7,7,6]
noise = np.random.uniform(-5,12,size=(10,3))
data = np.vstack([C1,C2,noise])
df = pd.DataFrame(data, columns=['f1','f2','f3'])
df.head()

x=df[['f1','f2','f3']].values

scale=StandardScaler()
x_scaled=scale.fit_transform(x)


dbscn=DBSCAN(eps=0.5,min_samples=5)
df['cluster']=dbscn.fit_predict(x_scaled)

fig=plt.figure(figsize=(9,5))
ax= fig.add_subplot(projection='3d')
ax.scatter(df['f1'],df['f2'],df['f3'],c=df['cluster'],cmap='rainbow')
plt.show()

print('Silhoutte Score => '
      ,silhouette_score(x_scaled,df['cluster']))


print(df.groupby('cluster')[['f1','f2','f3']].mean())

print('------------Seperate-----------')


#6

np.random.seed(21)
amount = np.concatenate([np.random.normal(50,10,70), np.random.normal(300,30,25)])
frequency = np.concatenate([np.random.normal(2,0.5,70), np.random.normal(8,1.5,25)])
# add extreme noisy customers
extreme = np.array([[1000, 0.5], [5, 50], [900, 90]])
data = np.vstack([np.column_stack([amount, frequency]), extreme])
df = pd.DataFrame(data, columns=['amount','frequency'])
df.head()

x=df[['amount','frequency']].values
scale=StandardScaler()
x_scalled=scale.fit_transform(x)

dbscan=DBSCAN(eps=0.5,min_samples=4)
df['cluster']=dbscan.fit_predict(x_scalled)

plt.scatter(df['amount'],df['frequency'],c=df['cluster'],cmap='rainbow')
plt.show()

print(df['cluster'].value_counts())

print('-- Silhoutte Score => ',silhouette_score(x_scalled,df['cluster']))

print(df.groupby('cluster')[['amount','frequency']].mean())

'''
-- Cluster 0 ==> Lowest amount and frequency 
-- Cluster 1 ==> avarege amount and frequency 
-- Outliers(-1) ==> higest amount and frequency 
'''
print("consistent clients ==> ",
   df['amount'].max() - df['frequency'].max()  )

print('------------Seperate-----------')
#7
np.random.seed(33)
# grid-like clusters
grid = []
for i in [1,4,7]:
    for j in [1,4,7]:
        pts = np.random.randn(30,2)*0.25 + [i,j]
        grid.append(pts)
grid = np.vstack(grid)
# anomalies
anom = np.array([[10,10],[ -1, 8],[8,-1]])
data = np.vstack([grid, anom])
df = pd.DataFrame(data, columns=['x','y'])
df.head()

x=df[['x','y']].values

scale=StandardScaler()
x_scalled=scale.fit_transform(x)

dbscan=DBSCAN(eps=0.5,min_samples=5)
df['cluster_DBS']=dbscan.fit_predict(x_scalled)

plt.scatter(df['x'],df['y'],c=df['cluster_DBS'],cmap='rainbow')
plt.show()

print('Silhoutte Score => ',silhouette_score(x_scalled,df['cluster_DBS']))

print(df['cluster_DBS'].value_counts())

print(df.groupby('cluster_DBS')[['x','y']].mean())

print('------------Seperate-----------')
#================================================
# PCA 
#================================================

X, y = make_classification(n_samples=300, n_features=10, n_informative=4,
                           n_redundant=2, n_clusters_per_class=1, random_state=42)

df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
df['target'] = y

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df.drop(columns=['target']))

pca = PCA(n_components=None, random_state=42)
pca.fit(X_scaled)

explained_ratio = pca.explained_variance_ratio_
cum_explained = np.cumsum(explained_ratio)

# Print explained variance ratios
for i, (r, c) in enumerate(zip(explained_ratio, cum_explained), start=1):
    print(f"PC{i}: explained_variance_ratio = {r:.3f}, cumulative = {c:.3f}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.bar(range(1, len(explained_ratio)+1), explained_ratio)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree plot')

plt.subplot(1,2,2)
plt.plot(range(1, len(cum_explained)+1), cum_explained, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.axhline(y=0.90, color='r', linestyle='--', label='90% explained')
plt.legend()
plt.title('Cumulative Explained Variance')
plt.show()

n_comp = np.argmax(cum_explained >= 0.90) + 1
print(f"\nChoose n_components = {n_comp} to explain >= 90% variance")

# 6) PCA with chosen n_components
pca = PCA(n_components=n_comp, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print("\nPCA components shape:", X_pca.shape)

# 7) Visualization: plot first two principal components (colored by target)
plt.figure(figsize=(6,5))
scatter = plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', s=35, alpha=0.8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Data projected to first 2 Principal Components')
plt.colorbar(scatter, label='target')
plt.grid(True)
plt.show()

# 8) Reconstruction: from PCA space back to original scaled space
X_approx_scaled = pca.inverse_transform(X_pca)   # shape (n_samples, n_components) -> back to (n_samples, n_features)
# Unscale to original feature units (approximation)
X_approx = scaler.inverse_transform(X_approx_scaled)

# compute reconstruction error (MSE) between original X and reconstructed X_approx
mse_reconstruction = mean_squared_error(X, X_approx)
print(f"\nReconstruction MSE (original scale): {mse_reconstruction:.4f}")

# 9) Inspect first PCA component loadings (to interpret which original features contribute)
loadings = pd.DataFrame(pca.components_.T, index=[f'feat_{i}' for i in range(X.shape[1])],
                        columns=[f'PC{i+1}' for i in range(n_comp)])
print("\nPCA loadings (first components):")
print(loadings.round(3))

print('----------Seperate----------')

"""
#1
np.random.seed(42)

X1 = np.random.randn(100) * 2 + 10
X2 = X1 * 0.5 + np.random.randn(100) * 0.5
X3 = X1 * -0.2 + X2 * 0.4 + np.random.randn(100) * 0.3

df = pd.DataFrame({'X1': X1, 'X2': X2, 'X3': X3})
df.head()

x=df[['X1','X2','X3']].values

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

pca=PCA(n_components=None,random_state=42)
pca.fit(x_scaled)

var_ratio=pca.explained_variance_ratio_
cummulative=np.cumsum(var_ratio)

plt.subplot(1,2,1)
plt.bar(range(1,len(var_ratio)+1),var_ratio)
plt.title("Prin..")

plt.subplot(1,2,2)
plt.plot(range(1,len(cummulative)+1),cummulative)
plt.axhline(y=0.95,c='r',label='95%')
plt.title('Cummulative Summary')
plt.legend()
plt.show()

n=np.argmax(cummulative >=0.95) +1

pca=PCA(n_components=n ,random_state=42)
xpc=pca.fit_transform(x_scaled)

scatter=plt.scatter(xpc[:,0],xpc[:,1],alpha=0.8)
plt.grid(True)
plt.colorbar(scatter, label='target')
plt.show()

x_approx_sc=pca.inverse_transform(xpc)
x_approx=scaler.inverse_transform(x_approx_sc)

print("-- MSE",mean_squared_error(x,x_approx))
#طلع  0.062 ضعيف اوى دا عادى ؟

print('----------Seperate----------')

np.random.seed(50)

cluster_1 = np.random.randn(80, 3) + [1, 1, 1]
cluster_2 = np.random.randn(80, 3) + [6, 6, 6]
cluster_3 = np.random.randn(80, 3) + [10, 2, 7]

data = np.vstack([cluster_1, cluster_2, cluster_3])
df = pd.DataFrame(data, columns=['a', 'b', 'c'])
df.head()

x=df[['a','b','c']].values

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

pca=PCA(n_components=None,random_state=42)
pca.fit(x_scaled)

var_ratio=pca.explained_variance_ratio_

cummulative=np.cumsum(var_ratio)
plt.subplot(1,2,1)
plt.bar(range(1,len(var_ratio)+1),var_ratio)
plt.subplot(1,2,2)
plt.plot(range(1,len(cummulative)+1),cummulative)
plt.axhline(y=0.95,linestyle='--',c='r')
plt.show()

n=np.argmax(cummulative>=0.95)+1
pca=PCA(n_components=n,random_state=42)
xpc=pca.fit_transform(x_scaled)

scatter=plt.scatter(xpc[:,0],xpc[:,1],cmap='coolwarm',alpha=0.8,s=50)
plt.colorbar(scatter,label='Target')
plt.show()