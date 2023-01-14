# RecommendationBasedonAmazonFoodReview
```
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

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
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
```
```
#Import the data set
df = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
```

```
df.head()
```

Split the data randomly into train and test dataset. ( For example split it in 70/30 ratio)
```
#Split the training and test data in the ratio 70:30
train_data, test_data = train_test_split(df_final, test_size = 0.3, random_state=0)

print(train_data.head(5))
```

Output
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
Output
```
Test data shape:  (6883, 3)
Train data shape:  (16058, 3)
```

Build Collaborative Filtering model
Model-based Collaborative Filtering: Singular Value Decomposition
```
df_CF = pd.concat([train_data, test_data]).reset_index()
df_CF.tail()
```

Output
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

ProductId 	7310172001 	7310172101 	7800648702 	B00004CI84 	B00004CXX9 	B00004RBDU 	B00004RBDZ 	B00004RYGX 	B00004S1C6 	B000052Y74 	... 	B009KAQZ9G 	B009KAQZIM 	B009KOHGEK 	B009KP6HBM 	B009LRLB6U 	B009LT26BC 	B009M2LUEW 	B009PCDDO4 	B009QEBGIQ 	B009RB4GO4
UserId 																					
A100WO06OQR8BQ 	0.0 	0.0 	0.0 	0.0 	0.0 	1.0 	0.0 	0.0 	0.0 	0.0 	... 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0
A106ZCP7RSXMRU 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	... 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0
A1080SE9X3ECK0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	... 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0
A10G136JEISLVR 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	... 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0
A11ED8O95W2103 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	... 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0 	0.0

















