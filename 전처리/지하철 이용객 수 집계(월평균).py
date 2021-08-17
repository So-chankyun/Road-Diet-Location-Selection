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
import seaborn as sns
from matplotlib import font_manager, rc
import re

font_path = r'C:/Users/user/NanumFontSetup_TTF_ALL/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path, size=18).get_name()
rc('font',family=font_name)


# %%
subway_station = pd.read_csv('./data/subway/CARD_SUBWAY_MONTH_201912.csv',encoding="EUC-KR")
subway_station


# %%
subway_station['사용일자'].unique()


# %%
len(subway_station['역명'].unique())


# %%
subway_pos = pd.read_csv('./data/subway/subway_crd_line_info-main/지하철역_좌표.csv')
subway_pos


# %%
subway_pos.info()


# %%
subway_pos = subway_pos.rename(columns={'역이름':'역명'})


# %%
regex = "\(.*\)|\s-\s.*"


# %%
# subway_pos_name = subway_pos['역명']
subway_station_name = subway_station['역명'].unique()

subway_station['역명']= pd.Series([re.sub(regex,'',row) for row in subway_station['역명']])


# %%
# inner join 진행 시 서로에게 없는 역들은 정보가 사라진다.
# join 진행 시 결측치들을 찾아보자.

ss = set(subway_station['역명'])
sp = set(subway_pos['역명'])
only_ss = ss-sp # subway_station에만 있는 역이름들이다.
only_sp = sp-ss # subway_pos에만 있는 역이름들이다.
display(len(only_ss))
display(len(only_sp))
display(len(ss & sp))
display(len(ss | sp))

# %% [markdown]
# 6월 한달 간의 지하철 역별 이용객 수이다.

# %%
subway_station_data1 = subway_station.merge(subway_pos,how='right',left_on='역명',right_on='역명')
display(subway_station_data1)
# 좌표는 있으나 승객 정보가 없는 경우이다.
# 차후에 데이터를 채워넣던가 해야할듯 싶다.
# display(only_sp)


# %%
msno.matrix(df=subway_station_data1,color=(0.3,0.4,0.3))
plt.show()


# %%
subway_station_data2 = subway_station.merge(subway_pos,how='left',left_on='역명',right_on='역명')
display(subway_station_data2)
# 승객 데이터는 있지만 좌표데이터가 없는 경우이다.
# display(only_ss)


# %%
# 데이터는 있으나 좌표가 없는 데이터들은 좌표를 채워줄 수 있도록 한다.
msno.matrix(df=subway_station_data2,color=(0.3,0.4,0.7))
plt.show()


# %%
# only_ss_data = subway_station[subway_station['역명'].isin(only_ss)]
# station = only_ss_data[['노선명','역명']]
# station.to_csv('have_to_get_station_pos.csv')


# %%
subway_station_data = subway_station.merge(subway_pos,left_on='역명',right_on='역명')
display(subway_station_data)


# %%
subway_station_sum = subway_station_data.groupby(['역명','x','y']).sum().reset_index()
subway_station_sum


# %%
subway_station_sum.drop(['사용일자','등록일자'],axis=1,inplace=True)


# %%
subway_station_sum['월평균승차총승객수'] = subway_station_sum['승차총승객수'] / 31
subway_station_sum['월평균하차총승객수'] = subway_station_sum['하차총승객수'] / 31


# %%
subway_station_sum.head()


# %%
subway_station_sum.to_csv('지하철 승하차 수(좌표포함).csv')


# %%
subway_station_sum[['승차총승객수','하차총승객수']].agg(['min','max','mean','std'])


# %%
# Create a base map
# 지하철 역의 분포를 보여줌....

m_5 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()
for idx, row in subway_station_sum.iterrows():
    if not math.isnan(row['x']) and not math.isnan(row['y']):
        mc.add_child(Marker([row['y'], row['x']],popup=row['역명'],tooltip=row['승차총승객수']))
m_5.add_child(mc)

# r,g,b,lime
# gradient = {100000:'lime',200000:'skyblue',300000:'red',400000:'green',500000:'blue'}
# Add a heatmap to the base map
HeatMap(data=subway_station_sum[['y', 'x']], radius=10).add_to(m_5)

# Display the map
m_5


