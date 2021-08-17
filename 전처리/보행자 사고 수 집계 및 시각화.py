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

font_path = r'C:/Users/user/NanumFontSetup_TTF_ALL/NanumGothic.ttf'
font_name = font_manager.FontProperties(fname=font_path, size=18).get_name()
rc('font',family=font_name)


# %%
pd_acc_data = pd.read_csv('./data/보행자사고/accident_17_19.csv')
pd_acc_data.head()


# %%
pd_acc_data['피해운전자 연령'].unique()

# %% [markdown]
# 아동 : 13세 이하<br>
# 노인 : 65세 이상<br>
# 필터링해서 객체에 저장해보자.<br>
# 일단 '세'라는 단어를 없애자. 이후 int type으로 변경

# %%
for idx, age in pd_acc_data.iterrows():
    if '세 이상' in age['피해운전자 연령']:
        pd_acc_data.loc[idx]['피해운전자 연령'] = age.replace('세 이상','')
    elif '세' in age:
        pd_acc_data.loc[idx]['피해운전자 연령'] = age.replace('세','')

pd_acc_data['피해운전자 연령'].unique()


# %%
len(pd_acc_data)


# %%
len(pd_acc_data[pd_acc_data['피해운전자 연령']=='미분류'])


# %%
pd_acc_data = pd_acc_data[pd_acc_data['피해운전자 연령']!='미분류']
pd_acc_data.head()


# %%
pd_acc_data[['사망자수','중상자수','경상자수','부상신고자수']]


# %%
pd_acc_data['총사고자수'] = pd_acc_data[['사망자수','중상자수','경상자수','부상신고자수']].sum(axis=1)


# %%
pd_acc_data['총사고자수'].unique()


# %%
pd_acc_data.drop(columns={'총사고수','총사망자수'},inplace=True)


# %%
pd_acc_data.head()


# %%
pd_acc_data[['사망자수','중상자수','경상자수','부상신고자수']].agg(['min','max'])


# %%
display(pd_acc_data['사망자수'].unique())
display(pd_acc_data['중상자수'].unique())
display(pd_acc_data['경상자수'].unique())
display(pd_acc_data['부상신고자수'].unique())


# %%
display(pd_acc_data.groupby('사망자수').size())
display(pd_acc_data.groupby('중상자수').size())
display(pd_acc_data.groupby('경상자수').size())
display(pd_acc_data.groupby('부상신고자수').size())


# %%
pd_acc_data['피해운전자 연령'] = pd_acc_data['피해운전자 연령'].astype(int)
pd_acc_data.info()


# %%
pd_acc_data_child = pd_acc_data[pd_acc_data['피해운전자 연령'] <= 13]
pd_acc_data_older = pd_acc_data[pd_acc_data['피해운전자 연령'] >= 65]
display(len(pd_acc_data_child))
display(len(pd_acc_data_older))
display(len(pd_acc_data_older)+len(pd_acc_data_child))
print(2132/len(pd_acc_data)*100)


# %%
pd_acc_data['피해운전자 연령'].agg(['min','max','mean','std'])


# %%
sns.histplot(data=pd_acc_data,x='피해운전자 연령')
plt.show()


# %%
# 어린이 버블맵

sns.set_palette("pastel")

# Create a base map
m_4 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

# def color_producer(val):
#     if val <= 13:
#         return 'red'
#     elif val >= 65:
#         return 'blue'
#     else:
#         return 'black'

# Add a bubble map to the base map
for i in range(0,len(pd_acc_data_child)):
    Circle(
        location=[pd_acc_data_child.iloc[i]['Y'], pd_acc_data_child.iloc[i]['X']],
        radius=5, color='red',
        tooltip=str(pd_acc_data_child.iloc[i]['피해운전자 연령'])+'세').add_to(m_4)

# Display the map
m_4


# %%
# 노인 버블맵

sns.set_palette("pastel")

# Create a base map
m_4 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

# def color_producer(val):
#     if val <= 13:
#         return 'red'
#     elif val >= 65:
#         return 'blue'
#     else:
#         return 'black'

# Add a bubble map to the base map
for i in range(0,len(pd_acc_data_older)):
    Circle(
        location=[pd_acc_data_older.iloc[i]['Y'], pd_acc_data_older.iloc[i]['X']],
        radius=5,
        color='blue',
        tooltip=str(pd_acc_data_older.iloc[i]['피해운전자 연령'])+'세').add_to(m_4)

# Display the map
m_4


# %%
# 어린이 히트맵

m_5 = folium.Map(location=[37.5665,126.9780], tiles='cartodbpositron', zoom_start=13)

mc = MarkerCluster()
for idx, row in pd_acc_data.iterrows():
    if not math.isnan(row['X']) and not math.isnan(row['Y']):
        mc.add_child(Marker([row['Y'], row['X']],tooltip=row['피해운전자 연령']))
m_5.add_child(mc)

# r,g,b,lime
# gradient = {100000:'lime',200000:'skyblue',300000:'red',400000:'green',500000:'blue'}
# Add a heatmap to the base map
HeatMap(data=pd_acc_data_child[['Y', 'X']], radius=5).add_to(m_5)

# Display the map
m_5


