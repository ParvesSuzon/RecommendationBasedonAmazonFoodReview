**Import Necessary Libraries**
```
import numpy as np
import pandas as pd
import math
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
#from sklearn.externals import joblib
import scipy.sparse
from scipy.sparse import csr_matrix
import warnings; warnings.simplefilter('ignore')
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/Downloads/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results we write to the current directory are saved as output.
```
**Importing the Dataset**
```
#Import the data set
df = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
```
**Display part of Dataset**
```
# see few rows of the imported dataset
df.tail()
```
**Output**
```
 	ProductId 	UserId 		Score
568449 	B001EO7N10 	A28KG5XORO54AY 	5
568450 	B003S1WTCU 	A3I8AFVPEE8KI5 	2
568451 	B004I613EE 	A121AA1GQV751Z 	5
568452 	B004I613EE 	A3IBEVCTXKNOH 	5
568453 	B001LR2CU2 	A3LGQPJCZVL9UC 	5
```
**Display number of rows and columns**
```
# Check the number of rows and columns
rows, columns = df.shape
print("No of rows: ", rows) 
print("No of columns: ", columns) 
```
**Output**
```
No of rows:  568454
No of columns:  3
```
**Display Data types**
```
#Check Data types
df.dtypes
```
**Output**
```
ProductId    object
UserId       object
Score         int64
dtype: object
```
**Check if there are any missing values present**
```
# Check for missing values present
print('Number of missing values across columns-\n', df.isnull().sum())
```
**There are no missing values with total records 568454**

**Display summary statistics of input data**
```
# Summary statistics of 'rating' variable
# Summary statistics of 'rating' variable
df[['Score']].describe().transpose()
```
**Output**
```
 	count 		mean 		std 		min 	25% 	50% 	75% 	max
Score 	568454.0 	4.183199 	1.310436 	1.0 	4.0 	5.0 	5.0 	5.0
```
**Display minimum and maximum ratings**
```
# find minimum and maximum ratings 

def find_min_max_rating():
    print('The minimum rating is: %d' %(df['Score'].min()))
    print('The maximum rating is: %d' %(df['Score'].max()))
    
find_min_max_rating() 
```
**Output**
```
The minimum rating is: 1
The maximum rating is: 5
```
**Ratings are on scale of 1 - 5**

**Display Distibution of ratings**
```
# Check the distribution of ratings 
with sns.axes_style('white'):
    g = sns.factorplot("Score", data=df, aspect=2.0,kind='count')
    g.set_ylabels("Total number of ratings")
```
**Output**
![image](https://user-images.githubusercontent.com/10477414/212487413-dded37f9-e5b7-479a-b08b-63cb70498b90.png)

**Display number of unique user id and product id in the data**
```
# Number of unique user id and product id in the data
print('Number of unique USERS in Raw data = ', df['UserId'].nunique())
print('Number of unique ITEMS in Raw data = ', df['ProductId'].nunique())
```
**Output**
```
Number of unique USERS in Raw data =  256059
Number of unique ITEMS in Raw data =  74258
```
**Take subset of dataset to make it less sparse/more dense. ( For example, keep the users only who has given 50 or more number of ratings)**
```
# Top 10 users based on rating
most_rated = df.groupby('UserId').size().sort_values(ascending=False)[:10]
most_rated
```
**Output**
```
UserId
A3OXHLG6DIBRW8    448
A1YUL9PCJR3JTY    421
AY12DBB0U420B     389
A281NPSIMI1C2R    365
A1Z54EM24Y40LL    256
A1TMAVN4CEM8U8    204
A2MUGFV2TDQ47K    201
A3TVZM3ZIXG8YW    199
A3PJZ8TU8FDQ1K    178
AQQLWCMRNDFGI     176
dtype: int64
```
**Data model preparation as per requirement on number of minimum ratings**
```
counts = df['UserId'].value_counts()
df_final = df[df['UserId'].isin(counts[counts >= 50].index)]
```
```
df_final.head()
```
**Output**
```
	ProductId 	UserId 		Score
14 	B001GVISJM 	A2MUGFV2TDQ47K 	5
44 	B001EO5QW8 	A2G7B7FKP2O2PU 	5
46 	B001EO5QW8 	AQLL2R1PPR46X 	5
109 	B001REEG6C 	AY12DBB0U420B 	5
141 	B001GVISJW 	A2YIO225BTKVPU 	4
```
```
print('Number of users who have rated 50 or more items =', len(df_final))
print('Number of unique USERS in final data = ', df_final['UserId'].nunique())
print('Number of unique ITEMS in final data = ', df_final['ProductId'].nunique())
```
**Output**
```
Number of users who have rated 50 or more items = 22941
Number of unique USERS in final data =  267
Number of unique ITEMS in final data =  11313
```

**df_final has users who have rated 50 or more items**

**Calculate the density of the rating matrix**
```
final_ratings_matrix = pd.pivot_table(df_final,index=['UserId'], columns = 'ProductId', values = "Score")
final_ratings_matrix.fillna(0,inplace=True)
print('Shape of final_ratings_matrix: ', final_ratings_matrix.shape)
given_num_of_ratings = np.count_nonzero(final_ratings_matrix)
print('given_num_of_ratings = ', given_num_of_ratings)
possible_num_of_ratings = final_ratings_matrix.shape[0] * final_ratings_matrix.shape[1]
print('possible_num_of_ratings = ', possible_num_of_ratings)
density = (given_num_of_ratings/possible_num_of_ratings)
density *= 100
print ('density: {:4.2f}%'.format(density))
```
**Output**
```
Shape of final_ratings_matrix:  (267, 11313)
given_num_of_ratings =  20829
possible_num_of_ratings =  3020571
density: 0.69%
```
**Split the data randomly into train and test dataset. 
( For example split it in 70/30 ratio)**
```
#Split the training and test data in the ratio 70:30
train_data, test_data = train_test_split(df_final, test_size = 0.3, random_state=0)

print(train_data.head(5))
```
**Output**
```
         ProductId          UserId  Score
399863  B002IEVJRY  A1N5FSCYN4796F      3
20262   B001BDDTB2  A1Q7A78VSQ5GQ4      5
139611  B001BCXTGS  A2PNOU7NXB1JE4      3
455504  B005HG9ERW  A2SZLNSI5KOQJT      3
512008  B0028PDER6   ALSAOZ1V546VT      5
```
```
def shape():
    print("Test data shape: ", test_data.shape)
    print("Train data shape: ", train_data.shape)
shape()
```
**Output**
```
Test data shape:  (6883, 3)
Train data shape:  (16058, 3)
```

**Build Collaborative Filtering model
Model-based Collaborative Filtering: Singular Value Decomposition**
```
df_CF = pd.concat([train_data, test_data]).reset_index()
df_CF.tail()
```

**Output**
```
	 index 	ProductId 	UserId 	        Score
22936 	275741 	B001M23WVY 	AY1EF0GOH80EK 	2
22937 	281102 	B002R8SLUY 	A16AXQ11SZA8SQ 	5
22938 	205589 	B00473PVVO 	A281NPSIMI1C2R 	5
22939 	303238 	B0002DGRZC 	AJD41FBJD9010 	5
22940 	36703 	B000EEWZD2 	A2M9D9BDHONV3Y 	3
```

```
#User-based Collaborative Filtering
# Matrix with row per 'user' and column per 'item' 
pivot_df = pd.pivot_table(df_CF,index=['UserId'], columns = 'ProductId', values = "Score")
pivot_df.fillna(0,inplace=True)
print(pivot_df.shape)
pivot_df.head()
```
Output
```
(267, 11313)
```

**SVD method**
**SVD is best to apply on a large sparse matrix**
```
from scipy.sparse.linalg import svds
# Singular Value Decomposition
U, sigma, Vt = svds(pivot_df, k = 50)
# Construct diagonal array in SVD
sigma = np.diag(sigma)
```

**Note that for sparse matrices, we can use the sparse.linalg.svds() function to perform the decomposition.**

SVD is useful in many tasks, such as data compression, noise reduction similar to Principal Component Analysis and Latent Semantic Indexing (LSI), used in document retrieval and word similarity in Text mining
```
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 

# Predicted ratings
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = pivot_df.columns)
preds_df.head()
```
```
# Recommend the items with the highest predicted ratings

def recommend_items(userID, pivot_df, preds_df, num_recommendations):
      
    user_idx = userID-1 # index starts at 0
    
    # Get and sort the user's ratings
    sorted_user_ratings = pivot_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_ratings
    sorted_user_predictions = preds_df.iloc[user_idx].sort_values(ascending=False)
    #sorted_user_predictions

    temp = pd.concat([sorted_user_ratings, sorted_user_predictions], axis=1)
    temp.index.name = 'Recommended Items'
    temp.columns = ['user_ratings', 'user_predictions']
    
    temp = temp.loc[temp.user_ratings == 0]   
    temp = temp.sort_values('user_predictions', ascending=False)
    print('\nBelow are the recommended items for user(user_id = {}):\n'.format(userID))
    print(temp.head(num_recommendations))
```
```
#Enter 'userID' and 'num_recommendations' for the user #
userID = 121
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)
```
```
Below are the recommended items for user(user_id = 121):

                   user_ratings  user_predictions
Recommended Items                                
B004E4EBMG                  0.0          1.553272
B004JGQ15E                  0.0          0.972833
B0061IUIDY                  0.0          0.923977
B0041NYV8E                  0.0          0.901132
B001LG940E                  0.0          0.893659
```

**Evaluate the model. ( Once the model is trained on the training data, it can be used to compute the error (RMSE) on predictions made on the test data.)**
**Evaluation of Model-based Collaborative Filtering (SVD)**

```
# Actual ratings given by the users
final_ratings_matrix.head()
```
```
# Average ACTUAL rating for each item
final_ratings_matrix.mean().head()
```
**Output**
```
ProductId
7310172001    0.037453
7310172101    0.037453
7800648702    0.018727
B00004CI84    0.044944
B00004CXX9    0.044944
dtype: float64
```
**Display average predicted rating for each item**
```
# Average PREDICTED rating for each item
preds_df.mean().head()
```
**Output**
```
ProductId
7310172001    0.001174
7310172101    0.001174
7800648702    0.004557
B00004CI84    0.039487
B00004CXX9    0.039487
dtype: float64
```

```
rmse_df = pd.concat([final_ratings_matrix.mean(), preds_df.mean()], axis=1)
rmse_df.columns = ['Avg_actual_ratings', 'Avg_predicted_ratings']
print(rmse_df.shape)
rmse_df['item_index'] = np.arange(0, rmse_df.shape[0], 1)
rmse_df.head()
```
**Output**
```
(11313, 2)
```
```
RMSE = round((((rmse_df.Avg_actual_ratings - rmse_df.Avg_predicted_ratings) ** 2).mean() ** 0.5), 5)
print('\nRMSE SVD Model = {} \n'.format(RMSE))
```
**Output**
```
RMSE SVD Model = 0.00995 
```

**Get top - K ( K = 5) recommendations. Since our goal is to recommend new products to each user based on his/her habits, we will recommend 5 new products.**

```
# Enter 'userID' and 'num_recommendations' for the user #
userID = 200
num_recommendations = 5
recommend_items(userID, pivot_df, preds_df, num_recommendations)
```
```
Below are the recommended items for user(user_id = 200):

                   user_ratings  user_predictions
Recommended Items                                
B004BKLHOS                  0.0          0.823791
B0061IUIDY                  0.0          0.622365
B004JRO1S2                  0.0          0.538305
B0061IUKDM                  0.0          0.534249
B000EQT77M                  0.0          0.529929
```


**Conclusion**

Model-based Collaborative Filtering is a personalised recommender system, the recommendations are based on the past behavior of the user and it is not dependent on any additional information.




