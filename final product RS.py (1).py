#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


from surprise import KNNBasic, SVD, NormalPredictor, KNNBaseline,KNNWithMeans, KNNWithZScore, BaselineOnly, CoClustering, Reader, dataset, accuracy


# In[3]:


columns = ['event_time', 'order_id','product_id','category_code','brand','price','user_id','rating']

df = pd.read_csv("C:\\Users\\vamsh\\OneDrive\\Documents\\MINI PROJECT II\\productname.csv",names=columns)


# In[4]:


df.tail(5)


# In[5]:


df.info()


# In[6]:


df.dtypes


# In[7]:


df['rating']=df['rating'].astype('float')


# In[8]:


df.shape


# In[9]:


df.describe().T


# In[10]:


#Dropping the "event_time,order_id,price" as it is not a needed field

df=df.drop(['event_time','order_id','price'],axis=1)


# In[11]:


df.dtypes


# In[12]:


#missing value
df.isna().sum()


# In[13]:


#dropping rows NaN values
df.dropna(inplace=True)


# In[14]:


df.shape


# In[15]:


#plot histogram
df.hist('rating',bins=15)


# In[16]:


popular = df[['user_id','rating']].groupby('user_id').sum().reset_index()
popular_20 = popular.sort_values('rating', ascending=False).head(n=15)
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt
 
objects = (list(popular_20['user_id']))
y_pos = np.arange(len(objects))
performance = list(popular_20['rating'])
 
plt.bar(y_pos, performance, align='center', alpha=0.8)
plt.xticks(y_pos, objects, rotation='vertical')
plt.ylabel('user_id')
plt.title('Most popular')
 
plt.show()


# In[17]:


# find unique user
df.user_id.value_counts()


# In[18]:


print('Number of unique users', len(df['user_id'].unique()))


# In[19]:


print('Number of unique products', len(df['product_id'].unique()))


# In[20]:


print('Number of unique ratings',df['rating'].unique())


# In[21]:


min_ratings=df[(df['rating']<=2.0)]


# In[22]:


print('Number of unique products rated low',len(min_ratings['product_id'].unique()))


# In[23]:


med_ratings = df[(df['rating'] > 2.0) & (df['rating'] < 4.0)]
print('Number of unique products rated medium',len(med_ratings['product_id'].unique()))


# In[24]:


high_ratings = df[(df['rating'] >= 4.0)]
print('Number of unique products rated high',len(high_ratings['product_id'].unique()))


# In[25]:


avg_rating_prod = df.groupby('product_id').sum() / df.groupby('product_id').count()
avg_rating_prod.drop(['brand','category_code'], axis=1,inplace =True)
print ('Top 15 highly rated products \n',avg_rating_prod.nlargest(15,'rating'))
avg_rating_prod.shape


# In[26]:


#Take a subset of the dataset to make it less sparse/ denser. ( For example, keep the users only who has given 50 or more number of ratings )
user_id = df.groupby('user_id').count()


# In[27]:


top_user = user_id[user_id['rating'] >= 4].index
topuser_ratings_df = df[df['user_id'].isin(top_user)]
#topuser_ratings_df.drop('productID', axis=1, inplace = True)
topuser_ratings_df.shape


# In[28]:


topuser_ratings_df.tail()


# In[29]:


topuser_ratings_df.sort_values(by='rating', ascending=False).head()


# In[30]:


#Keep data only for products that have 2 or more ratings
prodID = df.groupby('product_id').count()
top_prod = prodID[prodID['rating'] >= 2].index
top_ratings_df = topuser_ratings_df[topuser_ratings_df['product_id'].isin(top_prod)]
top_ratings_df.sort_values(by='rating', ascending=False).head()


# In[31]:


top_ratings_df.shape


# In[32]:


#Split the data randomly into train and test dataset. ( For example, split it in 70/30 ratio)


# In[33]:


from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(top_ratings_df, test_size = 0.30, random_state=0)
train_data.head()


# In[34]:


test_data.head()


# In[35]:


#Build Popularity Recommender model.
#Building the recommendations based on the average of all user ratings for each product.
train_data_grouped = train_data.groupby('product_id').mean().reset_index()
train_data_grouped.head()


# In[36]:


train_data_sort = train_data_grouped.sort_values(['rating', 'product_id'], ascending=False)
train_data_sort.head(15)


# In[37]:


train_data.groupby('product_id')['rating'].count().sort_values(ascending=False).head(10) 


# In[38]:


ratings_mean_count = pd.DataFrame(train_data.groupby('product_id')['rating'].mean()) 
ratings_mean_count['rating_counts'] = pd.DataFrame(train_data.groupby('product_id')['rating'].count())  
ratings_mean_count.head() 


# In[39]:


pred_df = test_data[['product_id','category_code','brand','user_id','rating']]


# In[40]:


pred_df.rename(columns = {'rating' : 'true_ratings'}, inplace=True)


# In[41]:


pred_df = pred_df.merge(train_data_sort, left_on='product_id', right_on = 'product_id')


# In[42]:


pred_df.rename(columns = {'rating' : 'predicted_ratings'}, inplace = True)
pred_df.head()


# In[43]:


import sklearn.metrics as metric
from math import sqrt
MSE = metric.mean_squared_error(pred_df['true_ratings'], pred_df['predicted_ratings'])
print('The RMSE value for Popularity Recommender model is', sqrt(MSE))


# In[44]:


#Build Collaborative Filtering model
import surprise
from surprise import KNNWithMeans
from surprise.model_selection import GridSearchCV
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split


# In[45]:


reader = Reader(rating_scale=(0.5, 5.0))


# In[46]:


#Converting Pandas Dataframe to Surpise format
data = Dataset.load_from_df(top_ratings_df[['user_id','product_id','rating']],reader)


# In[47]:


# Split data to train and test
from surprise.model_selection import train_test_split
trainset, testset = train_test_split(data, test_size=.3,random_state=0)
type(trainset)


# In[48]:


#Training the model
#KNNWithMeans

algo_user = KNNWithMeans(k=10, min_k=6, sim_options={'name': 'pearson_baseline', 'user_based': True})
algo_user.fit(trainset)


# In[49]:


#SVD

svd_model = SVD(n_factors=50,reg_all=0.02)
svd_model.fit(trainset)


# In[50]:


#Evaluate both the models. ( Once the model is trained on the training data, it can be used to compute the error (like RMSE) on predictions made on the test data.) You can also use a different method to evaluate the models.
#Popularity Recommender Model (RMSE)


# In[51]:


MSE = metric.mean_squared_error(pred_df['true_ratings'], pred_df['predicted_ratings'])
print('The RMSE value for Popularity Recommender model is', sqrt(MSE))


# In[52]:


#Collaborative Filtering Recommender Model (RMSE)

print(len(testset))
type(testset)


# In[53]:


#
#KNNWithMeans

# Evalute on test set
test_pred = algo_user.test(testset)
test_pred[0]


# In[54]:


# compute RMSE
accuracy.rmse(test_pred) #range of value of error


# In[55]:


test_pred = svd_model.test(testset)


# In[56]:


# compute RMSE
accuracy.rmse(test_pred)


# In[57]:


#Parameter tuning of SVD Recommendation system

from surprise.model_selection import GridSearchCV
param_grid = {'n_factors' : [5,10,15], "reg_all":[0.01,0.02]}
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3,refit = True)


# In[58]:


gs.fit(data)


# In[59]:


# get best parameters
gs.best_params


# In[60]:


# Use the "best model" for prediction
gs.test(testset)
accuracy.rmse(gs.test(testset))


# In[61]:


#Get top - K ( K = 5) recommendations. Since our goal is to recommend new products to each user based on his habits, we will recommend 5 new products.


# In[62]:


from collections import defaultdict
def get_top_n(predictions, n=5):
  
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


# In[63]:


top_n = get_top_n(test_pred, n=5)
# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])


# In[ ]:




