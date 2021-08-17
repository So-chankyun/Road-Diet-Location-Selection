# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import geopandas as gpd
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math
import missingno as msno
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as qsns
from matplotlib import font_manager, rc

font_path = r'C:/Users/user/NanumFontSetup_TTF_ALL/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path, size=18).get_name()
rc('font',family=font_name)

# %% [markdown]
# ## 버스 정류소 승하차 데이터 및 집계구 데이터 결합

# %%
bus_stop_500m = pd.read_csv('./data/전처리 필요/버스정류소_500m_집계구.csv',encoding="EUC-KR")
bus_stop_ctm = pd.read_csv('./bus_stop_ctm_with_pos.csv')
# bus_stop_ctm.head()


# %%
display(bus_stop_500m.columns)
display(bus_stop_ctm.columns)


# %%
bus_stop_ctm = bus_stop_ctm.drop(columns={'Unnamed: 0','X좌표','Y좌표'},axis=1)


# %%
bus_stop_ctm.head()


# %%
bus_stop_500m.head()


# %%
display(len(bus_stop_500m))
display(len(bus_stop_500m['표준ID'].unique()))
display(len(bus_stop_500m['TOT_REG_CD'].unique()))


# %%
bus_stop_data = bus_stop_500m.merge(bus_stop_ctm,left_on='표준ID',right_on='표준ID')
bus_stop_data.head()


# %%
bus_stop_data.info()


# %%
bus_stop_data[['TOT_REG_CD','ADM_CD']] = bus_stop_data[['TOT_REG_CD','ADM_CD']].astype(str)


# %%
bus_stop_data.info()


# %%
bus_stop_data_left = bus_stop_500m.merge(bus_stop_ctm,how='left',left_on='표준ID',right_on='표준ID')


# %%
# 집계구 데이터에는 데이터가 있는데, 승하차 데이터는 없는 경우
msno.matrix(df=bus_stop_data_left,color=(0.6,0.1,0.7))
plt.show()


# %%
bus_stop_data_right = bus_stop_500m.merge(bus_stop_ctm,how='right',left_on='표준ID',right_on='표준ID')


# %%
# 승하차 데이터는 있는데 집계구 파일에 데이터가 없는 경우는 없다.
# 즉, 집계구에 있는 모든 데이터와 matching된다.

msno.matrix(df=bus_stop_data_right,color=(0.6,0.7,0.7))
plt.show()


# %%
bus_stop_data.drop('역명',axis=1,inplace=True)


# %%
bus_stop_data.info()


# %%
bus_stop_data[bus_stop_data['표준ID'] == 1101053010006]


# %%
bus_stop_data[bus_stop_data['TOT_REG_CD']=='1101053010006']


# %%
bus_stop_count = bus_stop_data.groupby('TOT_REG_CD').size()
count = list()

for i, row in bus_stop_data.iterrows():
    tot_reg_cd = row.loc['TOT_REG_CD'] # 집계구 코드를 가져와서 저장.
    count.append(bus_stop_count.loc[tot_reg_cd])

bus_stop_data['정류장수'] = pd.Series(count)


# %%
bus_stop_data.head()


# %%
bus_stop_ride = bus_stop_data.groupby('TOT_REG_CD')['승차'].sum()
bus_stop_get_off = bus_stop_data.groupby('TOT_REG_CD')['하차'].sum()
count_ride = list()
count_get_off = list()

for i, row in bus_stop_data.iterrows():
    tot_reg_cd = row.loc['TOT_REG_CD'] # 집계구 코드를 가져와서 저장.
    count_ride.append(bus_stop_ride.loc[tot_reg_cd]/row['정류장수']/31)
    count_get_off.append(bus_stop_get_off.loc[tot_reg_cd]/row['정류장수']/31)

bus_stop_data['평균승차수'] = pd.Series(count_ride)
bus_stop_data['평균하차수'] = pd.Series(count_get_off)


# %%
bus_stop_data.head()


# %%
len(bus_stop_data)


# %%
bus_stop_data.to_csv('집계구별 정류장수 및 승하차 데이터.csv')


