#!/usr/bin/env python
# coding: utf-8

# # PREDICTION USING UNSUPERVISED ML

# # DONE BY:Isvariyashree.H

# # Collecting Data

# In[1]:


# Import libraries and dataset
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.datasets import load_iris


# In[2]:


# Create dataframe 
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[3]:


df


# In[4]:


# To see target 
iris.target


# # Data analysing

# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


# Add species type(target) to our dataframe
df['species'] = iris.target


# In[8]:


df.head(3)


# In[9]:


# To see the target names
target = iris.target_names
target


# In[10]:


# To see species name on dataframe
df['species name'] = df['species'].apply(lambda x: target[x])
df.head()


# In[11]:


# To see number of species on each type
df['species name'].value_counts()


# In[12]:


# Based on sepal values
plt.rcParams['figure.figsize'] = (12, 8)
fig, ax = plt.subplots()
scatter = ax.scatter(df['sepal length (cm)'], df['sepal width (cm)'], c = df['species'])
# produce a legend with the unique colors from the scatter
legend = ax.legend(*scatter.legend_elements(),loc="upper right", title="Species")
plt.xlabel('sepal length')
plt.ylabel('sepal width')
plt.title('Sepal values', fontsize = 12)
ax.add_artist(legend)
plt.show()


# In[13]:


# Based on petal values
plt.rcParams['figure.figsize'] = (12, 8)
fig, ax = plt.subplots()
scatter = ax.scatter(df['petal length (cm)'], df['petal width (cm)'], c = df['species'])
# produce a legend with the unique colors from the scatter
legend = ax.legend(*scatter.legend_elements(),loc="upper left", title="Species")
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.title('Petal values', fontsize = 12)
ax.add_artist(legend)
plt.show()


# In[14]:


# To see all the values based on species in boxplot 
f, axes = plt.subplots(2, 2)
sns.boxplot(  y="sepal length (cm)", x= "species name", data=df, ax=axes[0, 0])
sns.boxplot(  y="sepal width (cm)", x= "species name", data=df, ax=axes[0, 1])
sns.boxplot(  y="petal length (cm)", x= "species name", data=df,  ax=axes[1, 0])
sns.boxplot(  y="petal width (cm)", x= "species name", data=df, ax=axes[1, 1])
plt.show()


# # Data Wrangling

# In[15]:


# To see null values
df.isnull().sum()


# In[16]:


# To see visually
plt.figure(figsize = (8, 6))
sns.heatmap(df.isnull(), cmap = 'viridis')


# # Choosing the number of clusters

# In[18]:


# Import library
from sklearn.cluster import KMeans


# In[19]:


x = df.iloc[:, [0, 1, 2, 3]].values


# In[20]:


wcss = []
k_range = range(1, 11)
for i in k_range:
    km = KMeans(n_clusters=i)
    km.fit(x)
    wcss.append(km.inertia_)


# In[21]:


# plot the wcss values(elbow method)
plt.plot(k_range, wcss, c = 'red')
plt.xlabel('Number of clusters')
plt.ylabel('wcss')
plt.title('Elbow method', fontsize = 13)
plt.show()


# # Implement K-Means Clustering

# In[22]:


# Apply kmeans to data
km  = KMeans(n_clusters=3)
pred = km.fit_predict(x)
pred


# In[23]:


# To see cluster center
km.cluster_centers_


# In[25]:


# Visualising the clusters - On the second two columns
plt.scatter(x[pred == 0, 2], x[pred == 0, 3], c = 'red', label = 'setosa')
plt.scatter(x[pred == 1, 2], x[pred == 1, 3], c = 'blue', label = 'versicolour')
plt.scatter(x[pred == 2, 2], x[pred == 2, 3], c = 'orange', label = 'virginica')
# Plotting the centroids of the clusters
plt.scatter(km.cluster_centers_[:, 2], km.cluster_centers_[:,3], s = 100, c = 'black', marker = '*', label = 'Centroids')
plt.legend()
plt.show()


# In[ ]:




