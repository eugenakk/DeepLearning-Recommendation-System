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

