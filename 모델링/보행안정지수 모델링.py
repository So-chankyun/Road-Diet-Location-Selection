# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import libpysal as ps
import missingno as msno
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
import geopandas as gp
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc
import seaborn as sns
import warnings
import lightgbm as lgbm

mpl.rcParams['axes.unicode_minus'] = False
warnings.simplefilter('ignore')

font_path = r'C:/Users/user/NanumFontSetup_TTF_ALL/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path, size=18).get_name()
rc('font',family=font_name)


# %%
acc_data = pd.read_csv('./집계구_안전지수_최종.csv',encoding="EUC-KR")
seoul_shp = gp.read_file('./data/서울시 행정구역 및 집계구 정보(SHP)/서울시_집계구_4326.shp',encoding="EUC-KR")
# pop_data = pd.read_csv('./data/pop/19년 12월 집계구코드별 평균생활인구수.csv')


# %%
acc_data.info()


# %%
seoul_shp.info()


# %%
seoul_shp.to_crs(epsg=4326)
seoul_shp.crs

# %% [markdown]
# ### Column명 변경
# - 지하철(1km -> SUB_NUM
# - 지하철 거리점수 -> SUB_DIS_POINT
# - 버스정류장 -> BUS_NUM
# - 버스승하차 -> BUS_AVG
# - 승하차평균 -> SUB_AVG
# - 구별총생활 -> TOT_POP_GU
# - 구별총생_1 -> TOT_POP_GU_AVG
# - 사고수 -> ACC_NUM
# - 차선수 -> LOAD_NUM
# - 평균속도 -> SPEED_AVG
# - 표준편차 -> SPEED_STD
# - 혼잡수치 -> CONGESTION
# 
# TARGET VARIABLE : ACC_NUM

# %%
acc_data.columns


# %%
acc_data.rename(columns={'지하철(1km':'SUB_NUM','지하철 거리점수':'SUB_DIS_POINT'
                         ,'버스정류장':'BUS_NUM','버스승하차':'BUS_AVG','승하차평균':'SUB_AVG'
                         ,'구별총생활':'TOT_POP_GU','구별총생_1':'TOT_POP_GU_AVG','사고수n':'ACC_NUM'
                        , '차선수':'LOAD_NUM','평균속도':'SPEED_AVG','표준편차':'SPEED_STD'
                        , '혼잡수치':'CONGESTION'},inplace=True)      
acc_data.columns

# %% [markdown]
# ### Column 생성
# 1. 면적 당 상가 수 = 상가 수 / 면적
# 2. 면적 당 시설 수 = 시설 수(어린이집, 공공기관, 보육시설, SOC_NUM) / 면적
# 3. 면적 당 버스정류장 = 정류장수 / 면적
# 4. 면적 당 지하철 수 = 지하철 수 / 면적
# 5. 면적 당 생활인구 수 = 구별총생활인구수 평균 / 면적
# %% [markdown]
# ### 데이터 분포 확인 및 결측치 제거
# #### 1. 지하철 거리 점수

# %%
# standard scaler 적용
sns.histplot(data=acc_data,x='SUB_DIS_POINT',color='blue',kde=True, element='poly')
plt.show()

# %% [markdown]
# #### 2. 버스 승하차 평균
# - 왼쪽으로 치우친 형태의 분포를 띄고 있다. 
# - log함수를 이용하여 skew된 분포를 정규분포형태로 만들어주자.

# %%
sns.histplot(data=acc_data,x='BUS_AVG',element="poly",color='green',kde=True)
plt.show()

# %% [markdown]
# #### 3. 지하철 승하차 평균
# - 왼쪽으로 치우친 분포를 보이고 있다.
# - log함수를 이용하여 정규분포 형태를 만들어주자.

# %%
sns.histplot(data=acc_data,x='SUB_AVG',color='violet',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 4. 1km 내 지하철 역 수
# %% [markdown]
# - 일단 log 씌워주자

# %%
sns.histplot(data=acc_data,x='SUB_AVG',color='darkred',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 5. 구별총생활인구수 평균

# %%
sns.histplot(data=acc_data,x='TOT_POP_GU_AVG',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# - 사고수n, 차선수, 평균속도, 표준편차, 혼잡수치에 결측치가 존재한다.
# - 사고수n : 0으로 대체한다.
# - 차선수 : 1로 대체한다.
# - 평균속도 : 평균속도의 분포를 살펴보고 평균으로 대체할 수 있도록 한다.
# - 표준편차 : 혼잡수치의 산출에 필요한 데이터이므로 column을 제거하도록한다.
# - 혼잡수치 : 혼잡수치의 평균으로 대체할 수 있도록 한다.

# %%
msno.matrix(df=acc_data,color=(0.6,0.1,0.7))
plt.show()


# %%
acc_data.isnull().sum()

# %% [markdown]
# ### 사고 수와 혼잡지수의 교집합 계산

# %%
# 사고 수와 혼잡지수 둘 다 null이 아는 row들만 추출하여 모델을 생성
intersect_data = acc_data[~(acc_data['ACC_NUM'].isna() | acc_data['CONGESTION'].isna())]
prediction_data = acc_data[acc_data['ACC_NUM'].isna() & ~acc_data['CONGESTION'].isna()]
empty_data = acc_data[acc_data['ACC_NUM'].isna() & acc_data['CONGESTION'].isna()]
len(empty_data)


# %%
msno.matrix(df=intersect_data,color=(0.8,0.1,0.3))
plt.show()


# %%
intersect_data.isnull().sum()

# %% [markdown]
# #### 1. 사고수

# %%
avg_acc_data = acc_data['ACC_NUM'].agg(['min','max','mean','std','median'])
avg_acc_data


# %%
sns.histplot(data=acc_data,x='ACC_NUM',color='indigo',element='poly',kde=True)
plt.show()


# %%
Q1 = acc_data['ACC_NUM'].quantile(.25)
Q2 = acc_data['ACC_NUM'].quantile(.5)
Q3 = acc_data['ACC_NUM'].quantile(.75)
display(Q1)
display(Q2)
display(Q3)

# %% [markdown]
# 일단 2분위수로 대체한다.

# %%
# 2분위수로 대체
acc_data['ACC_NUM'].fillna(Q2,inplace=True)
acc_data['ACC_NUM'].isnull().sum()

# %% [markdown]
# #### 2. 차선수

# %%
load_agg = acc_data['LOAD_NUM'].agg(['min','max','mean','std'])
load_agg


# %%
sns.histplot(data=acc_data,x='LOAD_NUM',color='orange',element='poly',kde=True)
plt.show()


# %%
# 평균값으로 대체
acc_data['LOAD_NUM'].fillna(load_agg['mean'],inplace=True)
acc_data['LOAD_NUM'].isnull().sum()

# %% [markdown]
# #### 3. 차량 속도

# %%
speed_avg_agg = acc_data['SPEED_AVG'].agg(['min','max','mean','std'])
speed_avg_agg


# %%
# standard scaler
sns.histplot(data=acc_data,x='SPEED_AVG',color='red',element='poly',kde=True)
plt.show()


# %%
acc_data['SPEED_AVG'].fillna(speed_avg_agg['mean'],inplace=True)
acc_data['SPEED_AVG'].isnull().sum()

# %% [markdown]
# #### 4. 혼잡수치

# %%
congestion_agg = acc_data['CONGESTION'].agg(['min','max','mean','std'])
congestion_agg


# %%
# standard scaler 적용.
sns.histplot(data=acc_data,x='CONGESTION',color='cadetblue',element='poly',kde=True)
plt.show()

# %% [markdown]
# - 0보다 작은 수들은 0으로 대체한다.
# - 마찬가지로 minmaxscaler를 사용한다.
# - 음수값들을 0으로 대체하고 분포를 다시 살펴본 후 null값을 대체하자.

# %%
acc_data.loc[acc_data['CONGESTION'] < 0,'CONGESTION'] = 0
speed_avg_agg = acc_data['CONGESTION'].agg(['min','max','mean','std'])
speed_avg_agg


# %%
intersect_data.loc[intersect_data['CONGESTION']<0,'CONGESTION'] = 0
intersect_data['CONGESTION'].agg(['min','max'])


# %%
acc_data['CONGESTION'].fillna(speed_avg_agg['mean'],inplace=True)
acc_data['CONGESTION'].isnull().sum()


# %%
acc_data.isnull().sum()

# %% [markdown]
# ### Merge shp data and accident data

# %%
# 현재 TOT_REG_CD가 int type이므로 str type으로 변경해주도록 한다.
# 차후 merge를 하기 위해서이다.
intersect_data['TOT_REG_CD']= intersect_data['TOT_REG_CD'].astype(str)


# %%
# prepare dataset
data = intersect_data.merge(seoul_shp,left_on='TOT_REG_CD',right_on='TOT_REG_CD')
data.head()


# %%
data.columns


# %%
essential_col=['TOT_REG_CD','BUS_NUM','BUS_AVG','SUB_NUM','SUB_DIS_POINT','SUB_AVG'
               ,'TOT_POP_GU_AVG','LOAD_NUM','SPEED_AVG','CONGESTION','geometry','ACC_NUM']
required_data = data.loc[:,essential_col]
required_data.head()

# %% [markdown]
# ### Data 분포 변환
# %% [markdown]
# #### 1. 버스 이용객 평균

# %%
required_data.loc[:,'BUS_AVG'] = np.log1p(data['BUS_AVG'])
sns.histplot(data=required_data,x='BUS_AVG',color='skyblue',element="poly",kde=True)
plt.show()


# %%
required_data.loc[:,'BUS_AVG'] = np.log1p(data['BUS_AVG'])
sns.histplot(data=required_data,x='BUS_AVG',element="poly",color='green',kde=True)
plt.show()

# %% [markdown]
# #### 2. 지하철 이용객 평균

# %%
required_data.loc[:,'SUB_AVG'] = np.log1p(data['SUB_AVG'])
sns.histplot(data=required_data,x='SUB_AVG',color='violet',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 3. 구별 총 생활인구 평균

# %%
required_data.loc[:,'TOT_POP_GU_AVG'] = np.log1p(data['TOT_POP_GU_AVG'])
sns.histplot(data=required_data,x='TOT_POP_GU_AVG',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 4. 지하철 거리 점수

# %%
sd_scaler = StandardScaler()
mm_scaler = MinMaxScaler()


# %%
sub_dis_point = required_data.loc[:,'SUB_DIS_POINT'].values.reshape((-1,1))
required_data.loc[:,'SUB_DIS_POINT'] = pd.Series(sd_scaler.fit_transform(sub_dis_point).flatten())
sns.histplot(data=required_data,x='SUB_DIS_POINT',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 5. 속도 평균

# %%
sub_dis_point = required_data.loc[:,'SPEED_AVG'].values.reshape((-1,1))
required_data.loc[:,'SPEED_AVG'] = pd.Series(sd_scaler.fit_transform(sub_dis_point).flatten())
sns.histplot(data=required_data,x='SPEED_AVG',color='magenta',element="poly",kde=True)
plt.show()

# %% [markdown]
# #### 6. 혼잡지수

# %%
sub_dis_point = required_data.loc[:,'CONGESTION'].values.reshape((-1,1))
required_data.loc[:,'CONGESTION'] = pd.Series(sd_scaler.fit_transform(sub_dis_point).flatten())
sns.histplot(data=required_data,x='CONGESTION',color='orange',element="poly",kde=True)
plt.show()


# %%
center = seoul_shp.centroid
X = pd.Series(center.x)
Y = pd.Series(center.y)
required_data.loc[:,'lon'] = X
required_data.loc[:,'lat'] = Y
required_data.head()

# %% [markdown]
# ### 독립변수, 종속변수 설정

# %%
essential_col[1:-2]


# %%
#Prepare Georgia dataset inputs
s_y = required_data['ACC_NUM'].values.reshape((-1,1))
s_X = required_data[essential_col[1:-2]].values
u = required_data['lon']
v = required_data['lat']
s_coords = list(zip(u,v))

# %% [markdown]
# ### Train and Test set 분리

# %%
s_y = sample_data['구별총생활인구수(월평균)'].values.reshape((-1,1))
s_X = sample_data[['정류장수', '월평균승차수', '월평균하차수','lon','lat']].values
u = sample_data['lon']
v = sample_data['lat']
coords = list(zip(u,v))

# standardScaler적용

# scaler = StandardScaler()
# s_y = required_data['TOT_POP_GU_AVG']
# s_X = required_data[['정류장수', '월평균승차수', '월평균하차수','lon','lat']]

# train set과 test set으로 나누는 과정으로 보인다.
# 해당 모델에 교차 검증이 있는 것으로 알고 있다.

# test set은 20%, 
# x_train, x_valid, y_train, y_valid = train_test_split(s_X, s_y, test_size=0.2, shuffle=True,random_state=34)

# cal_u = x_train['lon'].values
# cal_v = x_train['lat'].values
# cal_coords = list(zip(cal_u,cal_v))

# pred_u = x_valid['lon'].values
# pred_v = x_valid['lat'].values
# pred_coords = list(zip(pred_u,pred_v))

# # 위도, 경도 column 제거
# # display(x_train.columns)
# x_train.drop(['lon','lat'],axis=1,inplace=True)
# x_valid.drop(['lon','lat'],axis=1,inplace=True)

# # array로 변환
# X_train = scaler.fit_transform(x_train.values)
# Y_train = scaler.fit_transform(y_train.values.reshape((-1,1)))
# X_valid = scaler.fit_transform(x_valid.values)
# Y_valid = scaler.fit_transform(y_valid.values.reshape((-1,1)))

# %% [markdown]
# ### Modeling

# %%
#Calibrate GWR model
gwr_selector = Sel_BW(s_coords, s_y, s_X,kernel='gaussian')
gwr_bw = gwr_selector.search(bw_min=2)
print(gwr_bw)
gwr_model = GWR(s_coords, s_y, s_X, gwr_bw,kernel='gaussian')
gwr_results = gwr_model.fit()


# %%
gwr_results.params[0:5]


# %%
gwr_results.localR2[0:10]

# %% [markdown]
# ### 모델링 결과 확인

# %%
gwr_results.summary()

# %% [markdown]
# ### 예측 모델생성

# %%
#Prepare Georgia dataset inputs
col = essential_col[1:-2]
col.append('lon')
col.append('lat')

s_y = required_data['ACC_NUM']
s_X = required_data[col]

x_train, x_valid, y_train, y_valid = train_test_split(s_X, s_y, test_size=0.2, shuffle=True,random_state=34)

cal_u = x_train['lon'].values
cal_v = x_train['lat'].values
cal_coords = list(zip(cal_u,cal_v))

pred_u = x_valid['lon'].values
pred_v = x_valid['lat'].values
pred_coords = list(zip(pred_u,pred_v))

# 위도, 경도 column 제거
# display(x_train.columns)
x_train.drop(['lon','lat'],axis=1,inplace=True)
x_valid.drop(['lon','lat'],axis=1,inplace=True)

# array로 변환
X_train = x_train.values
Y_train = y_train.values.reshape((-1,1))
X_valid = x_valid.values
Y_valid = y_valid.values.reshape((-1,1))


# %%
#Calibrate GWR model
gwr_selector = Sel_BW(cal_coords, Y_train, X_train,kernel='gaussian')
gwr_bw = gwr_selector.search(bw_min=2)
print(gwr_bw)
gwr_model = GWR(cal_coords, Y_train, X_train, gwr_bw,kernel='gaussian')
gwr_results = gwr_model.fit()


# %%
scale = gwr_results.scale
residuals = gwr_results.resid_response
# 
display(type(pred_coords))
# test data로 예측을 해보고 결과 저장
pred_results = gwr_model.predict(np.array(pred_coords), X_valid, scale, residuals)


# %%
np.corrcoef(pred_results.predictions.flatten(), Y_valid.flatten())
# [0][1]

# %% [markdown]
# ### 사고수 예측

# %%
prediction_data


# %%
prediction_data.loc[prediction_data['CONGESTION']<0,'CONGESTION'] = 0
prediction_data['CONGESTION'].agg(['min','max'])


# %%
# 현재 TOT_REG_CD가 int type이므로 str type으로 변경해주도록 한다.
# 차후 merge를 하기 위해서이다.
prediction_data['TOT_REG_CD']= prediction_data['TOT_REG_CD'].astype(str)


# %%
# prepare dataset
p_data = prediction_data.merge(seoul_shp,left_on='TOT_REG_CD',right_on='TOT_REG_CD')
p_data.head()


# %%
essential_col=['TOT_REG_CD','BUS_NUM','BUS_AVG','SUB_NUM','SUB_DIS_POINT','SUB_AVG'
               ,'TOT_POP_GU_AVG','LOAD_NUM','SPEED_AVG','CONGESTION','geometry','ACC_NUM']
p_required_data = p_data.loc[:,essential_col]
p_required_data.head()


# %%
center = seoul_shp.centroid
X = pd.Series(center.x)
Y = pd.Series(center.y)
p_required_data.loc[:,'lon'] = X
p_required_data.loc[:,'lat'] = Y
p_required_data.head()


# %%
essential_col[1:-2]


# %%
s_y = p_required_data['ACC_NUM'].values.reshape((-1,1))
s_X = p_required_data[essential_col[1:-2]].values
u = p_required_data['lon']
v = p_required_data['lat']
s_coords = list(zip(u,v))


# %%
required_data.shape


# %%
p_required_data.shape


# %%
scale = gwr_results.scale
residuals = gwr_results.resid_response

display(scale)
display(residuals)
pred_results = gwr_model.predict(np.array(s_coords), s_X, scale, residuals)


# %%
pred_results.predictions


