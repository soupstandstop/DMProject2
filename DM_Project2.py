#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[106]:


df_test.to_csv('/media/user/67a2dc9c-17dd-44b5-969d-d7ac6dcfcc33/Proposal/DM_P2_data_test.tsv', sep='\t', header=True, index=False)


# In[107]:


df_test = pd.read_csv('/media/user/67a2dc9c-17dd-44b5-969d-d7ac6dcfcc33/Proposal/DM_P2_data_test.tsv', sep='\t')


# In[94]:


df.to_csv('/media/user/67a2dc9c-17dd-44b5-969d-d7ac6dcfcc33/Proposal/DM_P2_data.tsv', sep='\t', header=True, index=False)


# In[95]:


df = pd.read_csv('/media/user/67a2dc9c-17dd-44b5-969d-d7ac6dcfcc33/Proposal/DM_P2_data.tsv', sep='\t')


# In[96]:


df


# In[108]:


import sklearn.datasets as datasets

df_target = df_test


# In[109]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier()
dtree.fit(df,df_target)


# In[110]:


from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydot
dot_data = StringIO()
export_graphviz(dtree, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
Image(graph.create_png())


# In[ ]:


get_ipython().system('jupyter nbconvert --to script DM_Project2.ipynb')

