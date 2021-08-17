import pandas as pd
import os

pop_data_list = os.listdir('./LOCAL_PEOPLE_201912')
i = 0
# 일별로 집계구코드 기준 생활인구를 합산한다.
def agg_pop(date):
    # print('./LOCAL_PEOPLE_201912/'+date)
    df = pd.read_csv('./LOCAL_PEOPLE_201912/'+date)

    group_data = df.groupby('집계구코드')['총생활인구수'].sum().sort_values(ascending=False)
    print('집계구코드 개수 : {}'.format(len(df['집계구코드'].unique())))
    sum_data = group_data.to_frame('구별총생활인구수')
    
    return sum_data

pop_data = pd.DataFrame()

# 일별로 만든 생활인구 집계 data를 concat한다.
for date in pop_data_list:
    pop_data = pd.concat([pop_data,agg_pop(date)],axis=0)

# 마지막으로 합계된 dataframe을 다시 groupby한 후 평균 column을 생성해주자.
data = pop_data.groupby('집계구코드')['구별총생활인구수'].sum().to_frame()
data['구별총생활인구수(월평균)'] = data['구별총생활인구수'] / 31
data.to_csv('19년 12월 집계구코드별 평균생활인구수.csv')
data.head()

    