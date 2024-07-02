import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X = np.array([[30,50000],[35,60000],[40,80000],[25,30000],[45,100000],
              [20,20000],[50,120000],[55,150000],[60,140000],[28,40000]])

kmeans = KMeans(n_clusters=3,random_state=0)
kmeans.fit(X)

lables = kmeans.labels_
centers = kmeans.cluster_centers_

plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c=lables,cmap="viridis",s=50,alpha=0.8)
plt.scatter(centers[:,0],centers[:,1],c='red',s=200,marker='X',label="Centroids")
plt.xlabel('Age')
plt.ylabel("Income")
plt.title("K-Means Clustering of Customers")
plt.legend()
plt.show()