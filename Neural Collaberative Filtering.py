'''
Deep Learning을 활용하여 Matrix Factorization을 하는 Neural Collaborative Filtering.

-Tensorflow로 BPR 을 구성할 것.
-Tensorflow로 Neural Collaborative Filtering을 구성할 것
'''

'exec(%matplotlib inline)'
import os
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Embedding
np.set_printoptions(5,)

####### 데이터 셋 가져오기
ROOT_URL = "https://pai-datasets.s3.ap-northeast-2.amazonaws.com/recommender_systems/movielens_100k/datasets/"

# 데이터 가져오기
ratings_path = get_file("100k_ratings.csv", ROOT_URL+"ratings.csv")
movies_path = get_file("100k_movies.csv",ROOT_URL+"movies.csv")
users_path = get_file("100k_users.csv", ROOT_URL+"users.csv")

ratings_df = pd.read_csv(ratings_path)
movies_df = pd.read_csv(movies_path)
users_df = pd.read_csv(users_path)

# 다섯개 데이터를 Random으로 가져옴
print("ratings_df의 크기 : ", ratings_df.shape)
ratings_df.sample(5, random_state=1)

############ K-Core Sampling을 통해 user_id, item_id 추려내기
like_df = ratings_df.copy()

threshold = 5
count = 0

while True:
    prev_total = len(like_df)
    print(f"{count}회차 데이터 수 : {prev_total:,}개")
    
    total_user_per_item = (
        like_df
        .groupby('item_id')['user_id']
        .count())
    over_item_ids = total_user_per_item[
        total_user_per_item>threshold].index
    
    total_item_per_user = (
        like_df
        .groupby('user_id')['item_id']
        .count())
    over_user_ids = total_item_per_user[
        total_item_per_user>threshold].index
    
    like_df = like_df[
        (like_df.user_id.isin(over_user_ids))
        &(like_df.item_id.isin(over_item_ids))]

    if prev_total == len(like_df):
        print("종료")
        break
    count += 1

    #Bayesian Personalized Ranking과 Neural Collaberative Filtering의 성능을 비교!
    # Hit RATIO 이용하기
    trains = []
tests = []

########## 평가기준 설정
for i, group_df in like_df.groupby('user_id'):
    # 마지막 직전은 Train_, 미자믹은 test_
    train_, test_ = group_df.iloc[:-1], group_df.iloc[-1:]
    trains.append(train_)
    tests.append(test_)
    
train_df = pd.concat(trains)
test_df = pd.concat(tests)

# user_id를 기준으로 정렬된 것을 무작위로 섞음
train_df = train_df.sample(frac=1)
test_df = test_df.sample(frac=1)


######### ################ hit ratio를 위한 데이터셋 구성하기
# #유저별 평가한 영화목록 구성하기
itemset_per_user = (
    ratings_df
    .groupby('user_id')
    ['item_id']
    .apply(frozenset)
)

total_items = set(ratings_df.item_id.unique())

# 유저가 평가하지 않은 영화목록 구성하기
notseen_itemset_per_user = total_items - itemset_per_user
notseen_itemset_per_user = notseen_itemset_per_user.apply(list)


hit_ratio_df = test_df.copy()

hit_ratio_df['not_seen_list'] = hit_ratio_df.user_id.apply(
    lambda x : random.choices(notseen_itemset_per_user[x],k=100))

hit_ratio_df = hit_ratio_df.drop('rating',axis=1)
hit_ratio_df.head()

############ Bayesian Personalized Ranking을 텐서프로우로 작성
from tensorflow.keras.layers import Input

#input 구성
user_id = Input(shape=(), name='user')
pos_item_id = Input(shape=(), name='positive_item')  # 고객이 구매한 item
neg_item_id = Input(shape=(), name='negative_item') # 고객이 구매하지 않은 item

# 임베딩 레이어 구성
from tensorflow.keras.layers import Embedding

num_user = ratings_df.user_id.max() + 1
num_item = ratings_df.item_id.max() + 1
num_factor = 30

user_embedding_layer = Embedding(num_user, num_factor, name='user_embedding')
item_embedding_layer = Embedding(num_item, num_factor+1, name='item_embedding')

pos_item_embedding = item_embedding_layer(pos_item_id)
neg_item_embedding = item_embedding_layer(neg_item_id)

from tensorflow.keras.layers import Concatenate
from tensorflow.keras import backend as K

user_embedding = user_embedding_layer(user_id)
one_embedding = K.ones_like(user_embedding[:,-1:])

user_embedding = Concatenate()([user_embedding, one_embedding])

from tensorflow.keras.layers import Dot
from tensorflow.keras.activations import sigmoid

pos_score = Dot(axes=(1,1,))([user_embedding, pos_item_embedding])
neg_score = Dot(axes=(1,1,))([user_embedding, neg_item_embedding])

diff_score = pos_score - neg_score

probs = sigmoid(diff_score)
from tensorflow.keras.models import Model

model = Model([user_id, pos_item_id, neg_item_id], probs)

l2_pos_item = pos_item_embedding**2
l2_neg_item = neg_item_embedding**2
l2_user = user_embedding**2
l2_reg = 0.0001

#값이 너무 크지 않도록 값을 조절해준다.
weight_decay = l2_reg * tf.reduce_sum(l2_pos_item + l2_neg_item + l2_user)

model.add_loss(weight_decay)


#모델 컴파일하기
from tensorflow.keras.optimizers import Adagrad
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

model.compile(Adagrad(1.), 
              loss=BinaryCrossentropy(),
              metrics=[BinaryAccuracy()])


###################################학습 데이터 구성하기
# 유저별 평가한 영화목록 구성하기
itemset_per_user = (
    train_df
    .groupby('user_id')
    ['item_id']
    .apply(frozenset)
)

total_items = set(train_df.item_id.unique())

#########################학습 데이터 구성하기
# 유저가 평가하지 않은 영화목록 구성하기
notseen_itemset_per_user = total_items - itemset_per_user
notseen_itemset_per_user = notseen_itemset_per_user.apply(list)

def get_bpr_dataset(train_df, notseen_itemset_per_user):
    batch_train_df = train_df.copy()
    batch_train_df = batch_train_df.sample(frac=1)
    batch_train_df['negative_item'] = batch_train_df.user_id.apply(
        lambda x : random.choice(notseen_itemset_per_user[x]))
    
    X = {
        "user":batch_train_df['user_id'].values,
        "positive_item":batch_train_df['item_id'].values,
        "negative_item":batch_train_df['negative_item'].values
    }
    y = np.ones((len(batch_train_df),1))
    
    return X, y

X,y = get_bpr_dataset(train_df, notseen_itemset_per_user)

######################## NEURAL COLLABERATIVE FILTERING 구성하기
from tensorflow.keras.layers import Input

user_id = Input(shape=(), name='user')
item_id =Input(shape=(), name='item')

num_user = ratings_df.user_id.max() + 1
num_item = ratings_df.item_id.max() + 1
num_factor = 32

user_embedding = Embedding(num_user, num_factor)(user_id) 
item_embedding = Embedding(num_item, num_factor)(user_id) 

# NCF Layers 구성하기!
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dense

concat_embedding = Concatenate()([user_embedding, item_embedding])

hidden1 = Dense(32, activation='relu')(concat_embedding)
hidden2 = Dense(16, activation='relu')(hidden1)
hidden3 = Dense(8, activation='relu')(hidden2)
probs = Dense(1, activation='sigmoid')(hidden3)

#모델 구성하기!

#입력값은 `user_id`와 `item_id`가 Pair로 들어가게 되고, 출력값은 `user_id`가 `item_id`를 선호할 확률이 계산됩니다. 

from tensorflow.keras.models import Model
model = Model([user_id, item_id], probs, name='NCF')
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy

model.compile(Adam(1e-3), 
              loss=BinaryCrossentropy(),
              metrics=[BinaryAccuracy()])

# 고객이 평가하지 않은, 구해마지 않은 영화군 정의하기
# 유저별 평가한 영화목록 구성하기
itemset_per_user = (
    train_df
    .groupby('user_id')
    ['item_id']
    .apply(frozenset)
)

total_items = set(train_df.item_id.unique())

# 유저가 평가하지 않은 영화목록 구성하기
notseen_itemset_per_user = total_items - itemset_per_user
notseen_itemset_per_user = notseen_itemset_per_user.apply(list)
def get_ncf_dataset(train_df, notseen_itemset_per_user,negative_ratio=4.):
    # 모든 rating은 기본적으로 Positive Sample
    positive_samples = train_df.copy()    
    
    # Negative Sampling 수행하기
    negative_samples = (train_df
                        .sample(frac=negative_ratio, replace=True)
                        .copy())
    negative_samples['item_id'] = negative_samples.user_id.apply(
        lambda x : random.choice(notseen_itemset_per_user[x]))
    
    # Positive와 Negative의 라벨을 지정해주기
    positive_samples['label'] = True    
    negative_samples['label'] = False
    batch_df = pd.concat([positive_samples, 
                          negative_samples])
    # 순서를 무작위로 섞기
    batch_df = batch_df.sample(frac=1)
    
    X = {
        "user":batch_df['user_id'].values,
        "item":batch_df['item_id'].values,
    }
    y = batch_df['label'].values
    
    return X, y

#모델 학습하기
epoch = 10
for i in range(1, epoch+1):
    print("{}th".format(i))
    X,y = get_ncf_dataset(train_df, notseen_itemset_per_user)
    model.fit(X, y, batch_size=1024, verbose=2) 

# 모델 평가하기
# BPR 대비 성능을 비교하기 위해 HIT RATIO 살펴본다.
hit = 0.
for i, row in tqdm(hit_ratio_df.iterrows()):
    user = np.array([row.user_id])
    seens = np.array([row.item_id])
    pos_scores = model.predict([user,seens])
    pos_scores = pos_scores[0,0]
    
    not_seens = np.array(row.not_seen_list)
    users = np.array([row.user_id]*len(not_seens))   
    neg_scores = model.predict([users,not_seens])
    
    if pos_scores > np.sort(neg_scores.flatten())[-10]:
        hit += 1

hit_ratio = hit / len(hit_ratio_df)        
print(f"hit ratio : {hit_ratio:.3f}")