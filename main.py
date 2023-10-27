#!/usr/bin/env python
# coding: utf-8

# # Real Estate Price Predictor

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


housedata = pd.read_csv(r"C:\Users\salma\Desktop\ML Harry bhai\project-1 real_estate/data.csv")


# ## Preprocessing

# In[3]:


housedata.shape


# In[4]:


housedata


# In[5]:


housedata['CHAS'].value_counts()


# housedata.hist(bins=50,figsize=(20,15))

# ## Train-test spliting

# In[6]:


# For learning only
# def train_test_split(data,test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_size = int(len(data)*test_ratio)
#     test_indices = shuffled[:test_size]
#     train_indices = shuffled[test_size:]

#     return data.iloc[train_indices], data.iloc[test_indices]


# In[7]:


# train_set, test_set = train_test_split(housedata,0.2)

# print(f"Rows in train set : {len(train_set)} \nRows in test set: {len(test_set)}")


# In[8]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housedata,test_size=0.2, random_state=42)

print(f"Rows in train set : {len(train_set)} \nRows in test set: {len(test_set)}")

# test_set['CHAS'].value_counts()


# In[9]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size =0.2 , random_state = 42)

for train_index, test_index in split.split(housedata, housedata['CHAS']):
    strat_train_set = housedata.loc[train_index]
    strat_test_set = housedata.loc[test_index]
    


# In[10]:


# strat_test_set['CHAS'].value_counts()
strat_train_set['CHAS'].value_counts()


# In[48]:


housing = strat_train_set.copy() # for training
#Housing ke andr trainning wlaa data aagya


# In[12]:


housing.describe()


# ## Corelating 

# In[13]:


corr_matrix = housedata.corr()


# In[14]:


corr_matrix['MEDV'].sort_values(ascending=False)


# In[15]:


from pandas.plotting import scatter_matrix
attribute  = ["MEDV", "RM", "ZN","LSTAT"]
scatter_matrix(housedata[attribute] , figsize = (12,8))


# In[16]:


housedata.plot(kind="scatter", x= "RM", y = "MEDV", alpha=0.5)


# In[17]:


housing = strat_train_set.drop("MEDV", axis =1) #bina label ke
housing_label = strat_train_set['MEDV'].copy()


# In[18]:


housing_label


# ## Creating a Pipeline

# In[19]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


# In[20]:


my_pipleline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scalar', StandardScaler())
])

housing_num = my_pipleline.fit_transform(housing)


# In[21]:


housing_num


# ## Selecting a desired model

# In[22]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
# model = LinearRegression()
# model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num, housing_label)


# In[23]:


some_data = housing.iloc[:5]
some_label = housing_label[:5]


# In[24]:


prepared_data = my_pipleline.transform(some_data)


# In[25]:


model.predict(prepared_data)


# In[26]:


list(some_label)


# ## Evaluating the model

# In[27]:


from sklearn.metrics import mean_squared_error

house_predictions = model.predict(housing_num)
mse = mean_squared_error(housing_label,house_predictions)
rmse = np.sqrt(mse)


# In[28]:


rmse # This model is overfit


# ## Using a better Evaluation technique - K Cross Validation

# In[29]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model,housing_num,housing_label,scoring='neg_mean_squared_error' , cv=10)
rmse_scores = np.sqrt(-scores)


# In[30]:


rmse_scores


# In[31]:


def print_scores(scores):
    print("Scores are: ", scores)
    print("Mean is: ", scores.mean())
    print("Standard dev. is: ", scores.std())    


# In[32]:


print_scores(rmse_scores)


# ## Using JobLib

# In[34]:


from joblib import  dump,load
dump(model, "real-estate.joblib")


# ## Testing the model on test data

# In[40]:


X_test = strat_test_set.drop('MEDV', axis=1)
Y_test = strat_test_set['MEDV'].copy()

X_prepared = my_pipleline.transform(X_test)
final_predict = model.predict(X_prepared)

test_mse = mean_squared_error(Y_test,final_predict)
test_rmse = np.sqrt(test_mse)


# In[47]:


test_rmse


# In[44]:


# print(list(Y_test))
# print(final_predict) #Comparing the values


# In[46]:


housing_num[0]


# In[ ]:




