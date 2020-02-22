'''
FACTORIZATION MACHINE 알고리즘 및 구현

'''

%matplotlib inline
import os
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from tqdm import tqdm
np.set_printoptions(5,)


#데이터 가져오기
ROOT_URL = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens_100k/datasets/"

# 데이터 가져오기
ratings_path = get_file("100k_ratings.csv", ROOT_URL+"ratings.csv")
movies_path = get_file("100k_movies.csv",ROOT_URL+"movies.csv")
users_path = get_file("100k_users.csv", ROOT_URL+"users.csv")

ratings_df = pd.read_csv(ratings_path)
movies_df = pd.read_csv(movies_path)
users_df = pd.read_csv(users_path)


data_df = pd.merge(ratings_df, movies_df, on="item_id")
data_df = pd.merge(data_df, users_df, on="user_id")
data_df.head()

#################### 단일 특징별 평균 선호도 파악하기
#연령 별 평균 평점 확인하기
(
    data_df
    .groupby('age')
    ['rating']
    .mean()
    .plot('bar')
)

plt.show()

#성별 평균 평점 확인하기
(
    data_df
    .groupby('gender')
    ['rating']
    .mean()
    .plot('bar')
)

plt.show()

#출시 연도별 평점 확인하기
(
    data_df
    .groupby('year')
    ['rating']
    .mean()
    .plot('bar',figsize=(20,5))
)

plt.show()


# 직업별 평균 평점 확인하기
(
    data_df.
    groupby(['occupation'])
    ['rating']
    .mean()
    .sort_values()
    .plot('bar', figsize = (15,5))
)

plt.show()


# 장르별 평균 평점 확인하기
genres =['Comedy','Action','Animation','Drama']

genre_rating = {}
for genre in genres:
    grouped = data_df[data_df[genre]==1]
    
    avg_rating = grouped.rating.mean()
    genre_rating[genre] = avg_rating
genre_rating = pd.Series(genre_rating)

genre_rating.plot('bar')
plt.show()

##################################특징 간의 상호관계 파악하기

#영화 장르와 유저의 직업군에 따른 선호도 비교!

genres = ['Comedy','Action','Animation','Drama']
occupations = ['lawyer','doctor','educator','writer','executive','homemaker']

rows = []
for genre in genres:
    for occupation in occupations:
        grouped = data_df[(data_df[genre]==1) & (data_df['occupation']==occupation)]
        avg_rating = grouped.rating.mean()
        rows.append((genre, occupation, avg_rating))
genre_occupation_df = pd.DataFrame(rows, columns=['genre','occupation','rating'])        

pivoted = genre_occupation_df.pivot_table('rating','occupation','genre')
sns.heatmap(pivoted.loc[occupations,genres])
plt.show()

#영화 장르와 유저의 연령군에 따른 선호도 비교!
genres = ['Comedy','Action','Animation','Drama']
ages = [2,3,4,5,6,7,8,9,10]

rows = []
for genre in genres:
    for age in ages:
        grouped = data_df[(data_df[genre]==1) & (data_df['age']==age)]
        avg_rating = grouped.rating.mean()
        rows.append((genre, age, avg_rating))
        
genre_age_df = pd.DataFrame(rows, columns=['genre','age','rating'])        
sns.heatmap(genre_age_df.pivot_table('rating','age','genre'))
plt.show()

################################ ############ 범주형 데이터와 FACTORIZATION MACHINE(FM)


#  범주형 데이터를 숫자로 표현하는 방법
# (1) ONE-HOT ENCODING!!!!

sample_df = data_df.sample(5,random_state=0)
sample_df.occupation
# 범주형 데이터로 활용할 컬럼들만 따로 이름을 저장합니다.
genre_names = movies_df.columns[3:]
cate_cols = ["user_id", "item_id", "year", "age", "gender", "occupation"] + list(genre_names)
print(cate_cols)
# 범주형 데이터는 아래와 같이 저장되어 있습니다.
data_df.loc[:, cate_cols].sample(5, random_state=1)
for col in cate_cols:
    data_df.loc[:, col] = data_df.loc[:, col].astype('category')

# 카테고리컬 특성의 값을 index로 대체합니다.
def cate2int(df):
    cate_sizes = {}
    for col_name in df.columns:
        if pd.api.types.is_categorical_dtype(df[col_name]):
            cate_sizes[col_name] = len(df[col_name].cat.categories)
            df.loc[:,col_name] = df[col_name].cat.codes
    return df, cate_sizes

data_df, cate_sizes = cate2int(data_df)

# 범주형 데이터가 다음과 같이 index로 변경되었습니다.
data_df.loc[:, cate_cols].sample(5, random_state=1)


################### FACTORIZATION MAchine의 아이디어
# 다중선형회귀 (mULTIPLE LINEAR REGRESSION)부분
row = data_df.loc[
    (data_df.user_id==195) 
    & (data_df.item_id==172), :]
row


############## LINEAR REGRESSION PART
from tensorflow.keras.layers import Layer

class LinearModel(Layer):
    """
    Linear Logit
    y = w0 + x1 + x2 + ...
    """
    def build(self, input_shape):
        self.b = self.add_weight(shape=(1,),
                                 initializer='zeros',
                                 trainable=True)
        super().build(input_shape)    
    
    def call(self, inputs, **kwargs):
        logits = tf.reduce_sum(inputs, axis=0)
        return logits


########################## FACTORIZATION MACHINE PART
from itertools import combinations

class FactorizationMachine(Layer):
    """
    Factorization Machine Layer
    """
    def call(self, inputs, **kwargs):
        logits = 0.
        for v_i, v_j in combinations(inputs, 2):
            logits += tf.tensordot(v_i, v_j, axes(1, 1))
        return logits

class FactorizationMachine(Layer):
    """
    Factorization Machine Layer
    """
    def call(self, inputs, **kwargs):
        # List of (# Batch, # Embed) -> (# Batch, # Features ,# Embed)
        inputs = tf.stack(inputs, axis=1) 

        logits = tf.reduce_sum(tf.square(tf.reduce_sum(inputs, axis=1)) - tf.reduce_sum(tf.square(inputs), axis=1), axis=1) /2
        
        return logits[: , None] # (# batch, 1)의 꼴이 되어야 함


######################## 임베딩 레이어와 커스텀 레이어로 FM구현
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Model

fm_size = 8
col_names = ["user_id", "item_id", "age", "year"]

inputs = []
fm_embeds = []
linear_embeds = []

# Embedding Part : 모든 칼럼에 대해 임베딩을 해줍니다. 
for col_name in col_names:
    input_category_dim = cate_sizes[col_name]        
    x = Input(shape=(), name=col_name)
    
    lr_out = Embedding(input_category_dim, 1)(x)
    fm_out = Embedding(input_category_dim, fm_size)(x)
    
    inputs.append(x)
    linear_embeds.append(lr_out)
    fm_embeds.append(fm_out)


# LR Model Part
lr_logits = LinearModel(name='lr')(linear_embeds)

# FM model part
fm_logits = FactorizationMachine(name='fm')(fm_embeds)

# LR 파트와 FM파트를 합쳐서 최종 모델을 만들어줍니다. 
pred = lr_logits + fm_logits
model = Model(inputs, pred, name='movielens')

####### MOVIE LENS 데이터를 이용해 FM 모델 만들기
data_df.loc[:, ["user_id", "item_id", "age", "year", "rating"]].sample(5)

X = data_df.loc[:, ["user_id", "item_id", "age", "year"]] 
y = data_df[['rating']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.1,
                                                    random_state=12345)



trainset = tf.data.Dataset.from_tensor_slices(
    ({k : v.values.astype(np.int32) 
      for k, v in X_train.iteritems()}, 
     y_train.values))

validset = tf.data.Dataset.from_tensor_slices(
    ({k : v.values.astype(np.int32) 
      for k, v in X_test.iteritems()}, 
     y_test.values))



# 텐서로 바꿔주기
trainset = tf.data.Dataset.from_tensor_slices(
    ({k : v.values.astype(np.int32) 
      for k, v in X_train.iteritems()}, 
     y_train.values))

validset = tf.data.Dataset.from_tensor_slices(
    ({k : v.values.astype(np.int32) 
      for k, v in X_test.iteritems()}, 
     y_test.values))

####### 모델에 학습
batch_size = 256 
num_epoch = 50

hist = model.fit_generator(
    (trainset
     .shuffle(batch_size*10)
     .batch(batch_size)),
    epochs=num_epoch,
    validation_data=validset.batch(batch_size*4))

