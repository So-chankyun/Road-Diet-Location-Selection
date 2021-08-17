# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt


# %%
df = pd.read_csv('./LOCAL_PEOPLE_20210730.csv',encoding='cp949')


# %%
df.head()


# %%
df.columns


# %%
group_data = df.groupby('집계구코드')['총생활인구수'].sum().sort_values(ascending=False)
sum_data = group_data.to_frame('구별총생활인구수')
sum_data.to_csv('total_pop_by_gu.csv')


# %%
os.listdir('.')


# %%
traffic_data = pd.read_excel("./data/traffic/2019년 12월 서울시 차량통행속도.xlsx")
traffic_data.head()

# %% [markdown]
# 링크아이디가 유일하게 결정된다.<br>
# 따라서 도로명 별로 말고 링크아이디 별로 산정된 내역도 보내주자.

# %%
display(len(traffic_data))
display(len(traffic_data['링크아이디'].unique()))


# %%
traffic_data.columns


# %%
mean = list()
var = list()
std = list()
for i in range(0,len(traffic_data)):
    mean_data = traffic_data.loc[i,'01시':'24시'].mean()
    var_data = traffic_data.loc[i,'01시':'24시'].var()
    std_data = traffic_data.loc[i,'01시':'24시'].std()
    mean.append(mean_data)
    var.append(var_data)
    std.append(std_data)


# %%
traffic_data['MEAN'] = mean
traffic_data['VAR'] = var
traffic_data['STD'] = std

# %% [markdown]
# 주말과 평일의 데이터의 차이가 있겠는가? 검증해볼 필요가 있다.

# %%
aggregate_data_by_linkId = traffic_data.groupby('링크아이디')[['MEAN','VAR','STD']].mean()
aggregate_data_by_linkId.info()


# %%
aggregate_data_by_linkId.to_excel('링크아이디별 집계 데이터(주말,주중 구분X).xlsx')


# %%
traffic_data['요일'].unique()


# %%
type(traffic_data['요일'])


# %%
weekDay = ['월','화','수','목','금']
weekEnd = ['토','일']

aggregate_weekDay_by_linkId = traffic_data[traffic_data['요일'].isin(weekDay)]
aggregate_weekEnd_by_linkId = traffic_data[traffic_data['요일'].isin(weekEnd)]

weekDay_agg_data = aggregate_weekDay_by_linkId.groupby(['링크아이디','시점명','종점명'])[['MEAN','VAR','STD']].mean()
weekEnd_agg_data = aggregate_weekEnd_by_linkId.groupby(['링크아이디','시점명','종점명'])[['MEAN','VAR','STD']].mean()

display(weekDay_agg_data.sort_values(by=['STD','VAR','MEAN']))
display(weekEnd_agg_data.sort_values(by=['STD','VAR','MEAN']))


# %%
weekDay_agg_data.info()


# %%
weekDay_agg_data.to_excel('링크아이디별 주중 집계 데이터.xlsx')
weekEnd_agg_data.to_excel('링크아이디별 주말 집계 데이터.xlsx')


# %%
sns.histplot(data=weekDay_agg_data,x='MEAN')
plt.show()


# %%
sns.histplot(data=weekEnd_agg_data,x='STD')
plt.show()


# %%
import numpy as np

print('-'*15+' 주중 속도 표준편차 사분위수 '+'-'*15)
print('Q1 : {}'.format(np.quantile(weekDay_agg_data['STD'],0.25)))
print('Q2 : {}'.format(np.quantile(weekDay_agg_data['STD'],0.5)))
print('Q3 : {}\n'.format(np.quantile(weekDay_agg_data['STD'],0.75)))

print('-'*15+' 주말 속도 표준편차 사분위수 '+'-'*15)
print('Q1 : {}'.format(np.quantile(weekEnd_agg_data['STD'],0.25)))
print('Q2 : {}'.format(np.quantile(weekEnd_agg_data['STD'],0.5)))
print('Q3 : {}'.format(np.quantile(weekEnd_agg_data['STD'],0.75)))


# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(20,5))
ax1 = sns.boxplot(ax=ax,x='STD',data=weekDay_agg_data, width=0.5)
plt.show()


# %%
fig, ax = plt.subplots(figsize=(20,5))
ax2 = sns.boxplot(ax=ax,x='STD',data=weekEnd_agg_data, width=0.5)


# %%



# %%
traffic_data.head()


# %%
traffic_data.to_csv('traffic_aggregate_data.csv')
aggregate_result = traffic_data.groupby('도로명')[['MEAN','VAR','STD']].mean()
aggregate_result.to_csv('traffic_aggregate_data_by_road_name.csv')


# %%
traffic_data.to_excel('traffic_aggregate_data.xlsx')
aggregate_result = traffic_data.groupby('도로명')[['MEAN','VAR','STD']].mean()
aggregate_result.to_excel('traffic_aggregate_data_by_road_name.xlsx')


# %%
display(traffic_data['STD'].min())
display(traffic_data['STD'].max())
display(traffic_data['STD'].mean())
display(traffic_data['STD'].var())


# %%
Q3 = traffic_data['STD'].quantile(.75)
print(Q3*1.5)


# %%
std_data = traffic_data['STD']


# %%
len(traffic_data)


# %%
std_data[std_data > 0].sort_values()


# %%
traffic_data.iloc[37192]


# %%
edit_std_data = traffic_data[traffic_data['기능유형구분'] != '도시고속도로']['STD']
a = edit_std_data[edit_std_data > 0].sort_values()
a


# %%
traffic_data.iloc[80905]


# %%
traffic_data['기능유형구분'].unique()


# %%
rank = std_data.sort_values()
std_not_nan = [row for row in std_data.iloc[:] if row != 0]
a= std_not_nan.sort()
print(a)


# %%
traffic_data.iloc[148797].to_frame()


# %%
print(len(std_data[std_data > Q3*1.5]))
len(std_data[std_data > Q3*1.5]) / len(traffic_data)*100


# %%
len(traffic_data)


# %%
import seaborn as sns
ax = sns.boxplot(x='STD',data=traffic_data)


# %%
ax = sns.histplot(data=traffic_data,x='STD')

# %% [markdown]
# 일별 총 row수가 1억2천 정도된다...<br>
# 각 column이 어떠한 의미를 가지는지 확인해보고 정제할 필요가 있다.<br>
# 일단 21.01-21.06월까지의 데이터를 정제하는 것을 해보도록 하자.

# %%
display(traffic_data.tail())
display(len(traffic_data))

# %% [markdown]
# 1. data 다운로드
# 2. 압축해제
# 3. 다운로드 받은 데이터들을 open하여 data 수정(코드 작성하여 프로그램 돌려야 할듯)
# 4. 수정 후 각 데이터들을 저장.
# ---
# - 다운로드 받는 코드
# - 압축해제 후 다시 저장하는 코드
# - 데이터를 수정 후 다시 저장하는 코드
# 마지막 두개 항목은 같이 작성해도 좋을 것 같다.

# %%
sns.heatmap(traffic_data.isnull())


