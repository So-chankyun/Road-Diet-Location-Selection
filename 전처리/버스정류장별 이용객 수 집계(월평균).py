# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ### 버스정류장 위치 확인

# %%
import pandas as pd
import missingno as msno
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import font_manager, rc

font_path = r'C:/Users/user/NanumFontSetup_TTF_ALL/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path, size=18).get_name()
rc('font',family=font_name)


# %%
bus_stop= gpd.read_file('./서울시_버스정류소_좌표데이터/서울시_버스정류소_좌표데이터.shp',encoding="EUC-KR")
bus_stop.head()


# %%
bus_stop.columns


# %%
msno.matrix(df=bus_stop,color=(0.6,0.1,0.7))
plt.show()


# %%
bus_stop.geometry.head()


# %%
bus_stop.crs


# %%
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
import math


# %%
# 버스 정류장 좌표 지도
# Create a map
m_3 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

# Add points to the map
mc = MarkerCluster()
for idx, row in bus_stop.iterrows():
    if not math.isnan(row['Y좌표']) and not math.isnan(row['X좌표']):
        mc.add_child(Marker([row['Y좌표'], row['X좌표']],tooltip=row['정류소명']))
m_3.add_child(mc)

# Display the map
m_3

# %% [markdown]
# ### 버스 정류장 이용객 수

# %%
bus_stop_ctm = pd.read_csv('./data/bus/2019년 버스노선별 정류장별 시간대별 승하차 인원 정보1.csv',encoding='EUC-KR')
bus_stop_ctm.head()


# %%
bus_stop_ctm['사용년월'].unique()


# %%
len(bus_stop['표준ID'].unique())


# %%
bus_stop_ctm = bus_stop_ctm[bus_stop_ctm['사용년월'] == 201912]
len(bus_stop_ctm)


# %%
display(len(bus_stop_ctm['표준버스정류장ID'].unique()))
not_n_ars = bus_stop_ctm[bus_stop_ctm['버스정류장ARS번호'] != '~']['버스정류장ARS번호'].unique()
display(len(not_n_ars))
not_b_bus= bus_stop_ctm[bus_stop_ctm['버스정류장ARS번호'] != '~']['표준버스정류장ID'].unique()
display(len(not_b_bus))


# %%
display(bus_stop_ctm['사용년월'].unique()) # 202101
display(bus_stop_ctm['등록일자'].unique()) # 20210203

# %% [markdown]
# 각 정류장의 이용객 수를 모두 더한 결과를 출력하도록 한다.(사용년월 기준 202101)

# %%
sample = bus_stop_ctm.groupby(['표준버스정류장ID','역명']).agg('sum').reset_index()
sample.head()

# %% [markdown]
# 승차, 하차 기준으로 승객수를 모두 합하자.<br>
# 이후 합한 데이터를 heatmap으로 표시할 수 있도록 하자.

# %%
get_off = ['시하차총승객수' in col for col in list(sample)]
ride = ['시승차총승객수' in col2 for col2 in list(sample)]
sample['하차'] = sample.iloc[:,get_off].sum(axis=1)
sample['승차'] = sample.iloc[:,ride].sum(axis=1)
sample.head()


# %%
sample = sample.rename(columns={'표준버스정류장ID':'표준ID'})


# %%
bs = set(bus_stop['표준ID'])
sam = set(sample['표준ID'])
only_bs = bs-sam
only_sam = sam-bs
display(len(only_bs)) # 버스 정류장 데이터에만 있는 역갯수
display(len(only_sam)) # 버스 승객수 데이터에만 있는 역갯수

# %% [markdown]
# ## 정류소 좌표데이터와 승하차 데이터 결합

# %%
# 정류장 좌표데이터는 있으나 승하차 데이터가 없음
# 차후에 채워야할 필요가 있음.

m_bus_stop1 = bus_stop.merge(sample,how='left',left_on='표준ID',right_on='표준ID').loc[:,['표준ID','역명','X좌표','Y좌표','승차','하차']]
m_bus_stop1


# %%
msno.matrix(df=m_bus_stop1,color=(0.6,0.1,0.9))
plt.show()


# %%
m_bus_stop2 = bus_stop.merge(sample,how='right',left_on='표준ID',right_on='표준ID').loc[:,['표준ID','역명','X좌표','Y좌표','승차','하차']]
m_bus_stop2


# %%
# 승객수 데이터는 있는데 정류장 좌표가 없음.
# 차후에 채워야할 필요가 있음.
msno.matrix(df=m_bus_stop2,color=(0.6,0.1,0.1))
plt.show()


# %%
m_bus_stop_not_exist_pos = m_bus_stop2[m_bus_stop2['표준ID'].isin(only_sam)]
display(m_bus_stop_not_exist_pos)
m_bus_stop_not_exist_pos.to_excel('have_to_get_pos.xlsx')


# %%
not_virtual = list()
m_bus_stop_not_exist_pos = m_bus_stop2[m_bus_stop2['표준ID'].isin(only_sam)]
for i,row in m_bus_stop_not_exist_pos.iterrows():
    if '가상' not in row['역명']:
        not_virtual.append(row['역명'])

len(not_virtual)
# not_virtual


# %%
# inner join

m_bus_stop = bus_stop.merge(sample,left_on='표준ID',right_on='표준ID').loc[:,['표준ID','역명','X좌표','Y좌표','승차','하차']]
m_bus_stop.head()


# %%
m_bus_stop['월평균승차수'] = m_bus_stop['승차']/31
m_bus_stop['월평균하차수'] = m_bus_stop['하차']/31
m_bus_stop.head()


# %%
display(len(m_bus_stop['표준ID'].unique()))
display(len(m_bus_stop))


# %%
m_bus_stop.info()


# %%
m_bus_stop[['승차','하차']].agg(['min','max','mean','std'])


# %%
m_bus_stop.to_csv('bus_stop_ctm_with_pos.csv')


# %%
sns.set_palette("pastel")

# Create a base map
m_4 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

def color_producer(val):
    if val <= 97700:
        return 'red'
    else:
        return 'blue'

# Add a bubble map to the base map
for i in range(0,len(m_bus_stop)):
    Circle(
        location=[m_bus_stop.iloc[i]['Y좌표'], m_bus_stop.iloc[i]['X좌표']],
        radius=5,
        color=color_producer(m_bus_stop.iloc[i]['승차']),
        tooltip=str(m_bus_stop.iloc[i]['승차'])+'명',
        popup=m_bus_stop.iloc[i]['역명']).add_to(m_4)

# Display the map
m_4


# %%
# Create a base map
# 승차

m_5 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()
for idx, row in m_bus_stop.iterrows():
    if not math.isnan(row['Y좌표']) and not math.isnan(row['X좌표']):
        mc.add_child(Marker([row['Y좌표'], row['X좌표']],popup=row['역명'],tooltip=row['승차']))
m_5.add_child(mc)

# r,g,b,lime
# gradient = {100000:'lime',200000:'skyblue',300000:'red',400000:'green',500000:'blue'}
# Add a heatmap to the base map
HeatMap(data=m_bus_stop[['Y좌표', 'X좌표']], radius=10).add_to(m_5)

# Display the map
m_5


