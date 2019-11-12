---
layout: post
title:  "Divisive Hierarchical Quantum Clustering with Max-cut"
author: AJ Rasmusson
date:   2019-11-09
mathjax: true
categories: unsupervised-machine-learning max-cut
front-image: "assets/images/output_13_0.png"
source-code: "https://github.com/ajrazander/Unsupervised-QML/blob/master/Max-cut_2%2B_divisive_clustering.ipynb"
commentary: This post has a mouthful of a title, but it's something I'm rather proud of. While messing around with max-cut
I wondered if there was a way to do more than simply split a dataset in two. Talking about this with some other physics
grad students, a friend mentioned hierarchical techniques where you recurcively split the data in half until all the data
are in their own clusters. Now, if you can cleverly stop this process part way, you've effectively turned the binary classifier
that is max-cut into a multi-nary (made up word) classify! How cool! :D Hope you enjoy this one as much as I did.
---

# 2+ Clustering with Max-cut
This notebook is an example of unsupervised learning on a quantum computer. The data used are from the iris data set. For faster run time, you will need an [IBMQ account](https://quantum-computing.ibm.com) to use their 32 qubit simulator.

In the [previous notebook](https://ajrazander.github.io/unsupervised-machine-learning/max-cut/2019/11/07/Quantum-Max-cut-vs-K-means.html), the max-cut problem was solved using QAOA, which gave a quantum speed up to unsupervised learning. Solving the max-cut problem is effectively a binary classifier. In this notebook, we focus on techniques that allow a binary classifier to cluster data into 2+ groups. To summarize this idea with a single question: How can a binary classifier, like QAOA solving the max-cut problem, be used to cluster data into more than 2 clusters?

## Divisive Hierarchical Quantum Clustering
One solution is to apply a divisive ("top-down") [hierarchial clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) method. This algorithm starts with the whole data set (the top) and slowly breaks it up until all data points are individual clusters (the bottom). For QAOA solving max-cut a divisive hierarchical clustering algorithm would execute as follows: (0) solve the max-cut problem on the entire dataset resulting in two child clusters, (1) solve the max-cut problem on each child cluster, repeat (1) until every data point is in an individual cluster.

Let's get to it!


```python
import numpy as np
import pandas as pd
from numba import jit
from sklearn.datasets import load_iris

import matplotlib.pyplot as plt
import seaborn as sns

# Load IBM quantum computing options
from qiskit import IBMQ
IBMQ.load_account()  # Load account from disk

# Quantum Computing packages
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.translators.ising import max_cut
from qiskit.aqua.components.optimizers import COBYLA
```


```python
# Import Iris dataset
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['species'] = pd.Categorical.from_codes(iris_data.target, iris_data.target_names)
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>



The iris dataset contains 150 entries. For reasonable execution time on the simulated quantum computer, we'll need to chop this down to ~12 entries.


```python
# Remove species labels
data_full = df.loc[:,['petal length (cm)','petal width (cm)','sepal length (cm)','sepal width (cm)']]

# Reduce number of data points
df_sub = df[::13]  # Dataframe with species labels
data_sub = data_full[::13]  # Dataframe without species lables
print(len(data_sub),'data entries')
```

    12 data entries


Now that the data is all squared away, let's solve the max-cut problem with QAOA just as we did in the [previous notebook](https://github.com/ajrazander/Unsupervised-QML/blob/master/Max-cut.ipynb) except this time we want to solve the max-cut problem on each resulting child cluster. To do this, we will track how the data is cut each iteration. This gets a bit messing with for loops and if statements.

Each max-cut creates two new branches, resulting in a binary tree. Since the end case has all n data points in their own clusters, there must be n leaves. For a binary tree, that leaves (no pun intended) the minimum height of the tree $$h = \log_2{n}$$. Thus, at least $$h$$ iterations need to be completed.


```python
# Helper function(s)

# Computes pairwise L2-norms (@jit gives x10 speed up)
@jit(nopython=True)
def calc_w(data_array):
    n_instances = data_array.shape[0]
    w = np.zeros((n_instances, n_instances))
    for i in range(0,n_instances):
        for j in range(0,n_instances):
            w[i, j] = np.linalg.norm(data_array[i]-data_array[j])
    return w
```


```python
# This can take several minutes to run

# Minimum iterations to turn datapoints into their own clusters (i.e. min. height of binary tree)
h = int(np.log2(len(data_sub)))

# Max number of qubits your computer can simulate
comp_qubits = 15

# Copy data for manipulation
data = data_sub.copy()

# QAOA hyperparameters and backend initialization
p = 1  # Number of adiabatic steps must be > 0
optimizer = COBYLA()  # Arbitrary selection
provider = IBMQ.get_provider(group='open')  # Load provider to access backends
backend_ibm = provider.get_backend('ibmq_qasm_simulator')  # Simulate on IBM's cloud service
backend_local = BasicAer.get_backend('statevector_simulator')  # Simulate on local machine

# Iterate over minimum height of tree
for i in range(0,h):
    # Initialize 'labels' column for future QAOA output
    data.loc[:,'cluster_'+str(i)] = np.nan
    data.loc[:,'cut_'+str(i)] = np.nan
    
    # Select subsets of data based on clustering from previous max-cut solution
    dfs = []
    if i > 0:
        cluster_range = data.loc[:,'cluster_'+str(i-1)].unique()
        for j in cluster_range:
            df_cluster = data.loc[data['cluster_'+str(i-1)] == j,data.columns[:4]]
            # if df_cluster length is 1 then it can't be further cut, so only consider lengths > 1
            if len(df_cluster.index) > 1:
                dfs.append(df_cluster)
    else:
        dfs.append(data[data.columns[:4]])

    # Solve max-cut with QAOA on each child cluster
    for j, df_part in enumerate(dfs):
        w = calc_w(df_part.values)  # Calculate pairwise distances between points
        
        # Initialize QAOA and execute
        qubit_ops, offset = max_cut.get_max_cut_qubitops(w)
        qaoa = QAOA(qubit_ops, optimizer, p)
        # If there are 'too many' qubits, use IBM's 32 qubit backend
        if w.shape[0] > comp_qubits:
            backend = backend_ibm
        else:
            backend = backend_local
        quantum_instance = QuantumInstance(backend, shots=1, skip_qobj_validation=False)
        result = qaoa.run(quantum_instance)

        # Extract results
        x = max_cut.sample_most_likely(result['eigvecs'][0])

        # Store cluster results back into Dataframe. Labels must be unqiue each iteration hence + 2*j
        df_part.loc[:,'cluster_'+str(i)] = max_cut.get_graph_solution(x) + 2*j
        df_part.loc[:,'cut_'+str(i)] = max_cut.max_cut_value(x, w)
        
        # Update Dataframe with new clusters and cut weights
        data.update(df_part)

    print('Iteration',i+1,'of',h,'completed')
```

    Iteration 1 of 3 completed
    Iteration 2 of 3 completed
    Iteration 3 of 3 completed


Let's take a look at how the clustering compares to the known species labeling.


```python
# Include results from QAOA in df_sub dataframe for comparison to species label
for i in range(0,h):
    df_sub.loc[:,'cluster_'+str(i)] = data.loc[:,'cluster_'+str(i)]
    df_sub.loc[:,'cut_'+str(i)] = data.loc[:,'cut_'+str(i)]
df_sub
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
      <th>species</th>
      <th>cluster_0</th>
      <th>cut_0</th>
      <th>cluster_1</th>
      <th>cut_1</th>
      <th>cluster_2</th>
      <th>cut_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1.0</td>
      <td>150.460827</td>
      <td>1.0</td>
      <td>2.974861</td>
      <td>1.0</td>
      <td>0.561177</td>
    </tr>
    <tr>
      <th>13</th>
      <td>4.3</td>
      <td>3.0</td>
      <td>1.1</td>
      <td>0.1</td>
      <td>setosa</td>
      <td>1.0</td>
      <td>150.460827</td>
      <td>0.0</td>
      <td>2.974861</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>5.0</td>
      <td>3.4</td>
      <td>1.6</td>
      <td>0.4</td>
      <td>setosa</td>
      <td>1.0</td>
      <td>150.460827</td>
      <td>1.0</td>
      <td>2.974861</td>
      <td>0.0</td>
      <td>0.561177</td>
    </tr>
    <tr>
      <th>39</th>
      <td>5.1</td>
      <td>3.4</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
      <td>1.0</td>
      <td>150.460827</td>
      <td>1.0</td>
      <td>2.974861</td>
      <td>1.0</td>
      <td>0.561177</td>
    </tr>
    <tr>
      <th>52</th>
      <td>6.9</td>
      <td>3.1</td>
      <td>4.9</td>
      <td>1.5</td>
      <td>versicolor</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>2.0</td>
      <td>30.857643</td>
      <td>3.0</td>
      <td>3.252945</td>
    </tr>
    <tr>
      <th>65</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>4.4</td>
      <td>1.4</td>
      <td>versicolor</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>2.0</td>
      <td>30.857643</td>
      <td>3.0</td>
      <td>3.252945</td>
    </tr>
    <tr>
      <th>78</th>
      <td>6.0</td>
      <td>2.9</td>
      <td>4.5</td>
      <td>1.5</td>
      <td>versicolor</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>2.0</td>
      <td>30.857643</td>
      <td>2.0</td>
      <td>3.252945</td>
    </tr>
    <tr>
      <th>91</th>
      <td>6.1</td>
      <td>3.0</td>
      <td>4.6</td>
      <td>1.4</td>
      <td>versicolor</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>2.0</td>
      <td>30.857643</td>
      <td>2.0</td>
      <td>3.252945</td>
    </tr>
    <tr>
      <th>104</th>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.8</td>
      <td>2.2</td>
      <td>virginica</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>3.0</td>
      <td>30.857643</td>
      <td>5.0</td>
      <td>4.912491</td>
    </tr>
    <tr>
      <th>117</th>
      <td>7.7</td>
      <td>3.8</td>
      <td>6.7</td>
      <td>2.2</td>
      <td>virginica</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>3.0</td>
      <td>30.857643</td>
      <td>4.0</td>
      <td>4.912491</td>
    </tr>
    <tr>
      <th>130</th>
      <td>7.4</td>
      <td>2.8</td>
      <td>6.1</td>
      <td>1.9</td>
      <td>virginica</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>3.0</td>
      <td>30.857643</td>
      <td>4.0</td>
      <td>4.912491</td>
    </tr>
    <tr>
      <th>143</th>
      <td>6.8</td>
      <td>3.2</td>
      <td>5.9</td>
      <td>2.3</td>
      <td>virginica</td>
      <td>0.0</td>
      <td>150.460827</td>
      <td>3.0</td>
      <td>30.857643</td>
      <td>5.0</td>
      <td>4.912491</td>
    </tr>
  </tbody>
</table>
</div>



Solving the max-cut problem gives clustering labels as a binary tree. To see this, look at the columns 'cluster_(i)'. As you follow from one column to the next you see, for example, cluster 0 in 'cluster_0' is split into two clusters in column 'cluster_1' and so on.

## Stop Criteria 
To find the optimal clustering, a depth-first-search is performed down the tree. The search is stopped once the leaves are below some predefined cut weight. Below, the cut weights vs the associated number of clusters is plotted. We see (knowing the actual species) the best option is to follow the elbow rule and stop at 3 clusters. However, which of the two possible 3 cluster configurations do we choose? The one with the higher cut weight. The higher cut weight implies a more separated graph resulted from the cut, and the more the separation, the more likely the clusters are in fact different.


```python
# Plot max-cut weights vs number of clusters created
cuts = []
cluster_num = []

# Get max-cut weights and associated number of clusters from results in df_sub
for i in range(0,h):
    # Collect max-cut weights
    cuts += list(df_sub['cut_'+str(i)].unique())
    # Number how many clusters have been made for this cut
    cut_off = 2**i
    for j in range(0,cut_off):
        cluster_num.append(i+2)

plt.scatter(cluster_num,cuts)
plt.xlabel('Number of clusters')
plt.ylabel('Max cut weight')
plt.show()
```


![png](assets/images/output_11_0.png)


Let's traverse the binary tree along the 'heaviest' branches and stop at the number of leaves associated with the elbow rule. Based on the plot above that would be 3 clusters where the 2nd and 3rd clusters are associated with the point (3,~30)


```python
# cut_off sets where the 'elbow' is in for the elbow rule
cut_off = 0.10  # 10% of global maximum is acceptable

# First cut should be the largest compared to subsequent cuts
global_maxim = df_sub['cut_0'].max()

# Initialize final clustering column
df_sub.loc[:,'final'] = np.nan

for (clus_col, cut_col) in zip(df_sub[df_sub.columns[5::2]],df_sub[df_sub.columns[6::2]]):
    # Find the maximum cut for this particular column of data
    maxim = df_sub[cut_col].max()
    if maxim > global_maxim*cut_off:
        df_sub['final'].update(df_sub[clus_col][df_sub[cut_col] == maxim])

# Constrain data to final clustering assignments
df_sub_plot = df_sub[['sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)','final']]

# Visualize clustering
sns.pairplot(data=df_sub_plot,hue='final',palette="husl",vars=df_sub.columns[:4])
plt.show()

# Display mean of cluster labels by species
print('\"Average\" label classification:')
print(df_sub.groupby(['species']).sum()['final'] / df_sub.groupby(['species']).count()['final'])
```


![png](assets/images/output_13_0.png)


    "Average" label classification:
    species
    setosa        1.0
    versicolor    2.0
    virginica     3.0
    Name: final, dtype: float64


The 'Average' label classification is the average cluster label (1, 2, or 3) for each species. Setosa is all in cluster 1; versicolor is all in cluster 2; and virginica is all in cluster 3. The divisive hierarchical quantum clustering did a great job!
