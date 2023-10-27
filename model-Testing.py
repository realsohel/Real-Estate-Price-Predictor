#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from joblib import  dump,load
model = load("real-estate.joblib")


# In[10]:


input =  np.array([[-10.43942006,  12.12628155, -1.12165014, -0.27288841, -1.42262747,
       -1.01979304, -2.31238772, 5.61111401, -1.0016859 , -1.5778192 ,
       -0.97491834,  0.41164221, -1.86091034]])


# In[11]:


model.predict(input)


# In[ ]:




