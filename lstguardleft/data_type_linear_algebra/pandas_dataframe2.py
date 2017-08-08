
'''
/*********************************************************************************************************
-- Title : [Py2.7] Pandas.DataFrame 조작 - 피벗, 그룹핑, 집계, 그룹연산
-- Reference : Python for Data Analysis
-- Key word : 피벗 pivot pivot_table 그룹핑 그룹 groupby stack unstack 카테고리 category fill_value 그룹연산 aggfunc     
              margins 크로스탭 crosstab fillna pandas dataframe 데이터프레임
-- 출처: http://dbrang.tistory.com/1006 [dBRang]
*********************************************************************************************************/
'''

# -*- coding: utf-8 -*-
 
import os, sys
import pandas as pd
import json

import numpy as np
 
# ********************************************
# -- 데이터 피벗(pivot_table) : 많이 쓰임!!!
# ********************************************
print("-" * 100 + "[3]") # ----- #
 
data = pd.read_excel("./data/sales-funnel.xlsx")
print (data.head(3))
 
# -- 명확화 위해 카테고리 변수화
data["Status"] = data["Status"].astype("category")
data["Status"].cat.set_categories(["won","pending","declined"], inplace=True)
data.head(3)
 
print("-" * 100 + "[3.1]") # ----- #
 
# -- 피벗팅
#    그룹핑시 문자 컬럼은 사라짐, 디폴트는 avg이기에..
print (pd.pivot_table(data, index=["Name"]))
print (pd.pivot_table(data, index=["Name", "Rep", "Manager"]))
 
print("-" * 100 + "[3.2]") # ----- #
 
# -- 그룹핑 후 원하는 컬럼만 보기
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price"]))
 
print("-" * 100 + "[3.3]") # ----- #
 
 
# ********************************************
# -- 그룹핑 시 집계합수 적용
# ********************************************
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price"], aggfunc=np.sum))
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price"], aggfunc=[np.mean, len]))
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price", "Price"], aggfunc=[np.mean, len]))
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price", "Quantity"], aggfunc=[np.mean, len]))
 
print("-" * 100 + "[3.4]") # ----- #
 
 
# ********************************************
# -- 세로줄에 대한 계층
# ********************************************
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price"], columns=["Product"], aggfunc=np.sum))
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price", "Quantity"], columns=["Product"], aggfunc=np.sum))
 
 
print("-" * 100 + "[3.5]") # ----- #
 
 
# ********************************************
# -- NULL 처리하기(fill_value)
# ********************************************
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price"], columns=["Product"], aggfunc=np.sum, fill_value=0))
print (pd.pivot_table(data, index=["Manager", "Rep"], values=["Price", "Quantity"], columns=["Product"], aggfunc=np.sum, fill_value=0))
 
print("-" * 100 + "[3.6]") # ----- #
 
print (pd.pivot_table(data, index=["Manager", "Rep", "Product"], values=["Price", "Quantity"], aggfunc=np.sum, fill_value=0))
 
print("-" * 100 + "[3.7]") # ----- #
 
 
# ********************************************
# -- 전체 집계 추가(margins)
# ********************************************
# -- 하단에 전체 집계 추가
print (pd.pivot_table(data, index=["Manager", "Rep", "Product"], values=["Price", "Quantity"], aggfunc=[np.sum, np.mean], fill_value=0))
print (pd.pivot_table(data, index=["Manager", "Rep", "Product"], values=["Price", "Quantity"], aggfunc=[np.sum, np.mean], fill_value=0, margins=True))
 
print("-" * 100 + "[3.7]") # ----- #
 
 
# ********************************************
# -- values별 다른 집계 처리
# ********************************************
print (pd.pivot_table(data, index=["Manager", "Status"], values=["Quantity", "Price"], columns=["Product"]
               , aggfunc={"Quantity": len, "Price": [np.sum, np.mean]}, fill_value=0))
 
print("-" * 100 + "[3.8]") # ----- #
 
# -- 원하는 결과만 추출
table = pd.pivot_table(data, index=["Manager", "Status"], columns=["Product"], values=["Quantity", "Price"]
               , aggfunc={"Quantity": len, "Price": [np.sum, np.mean]}, fill_value=0)
print (table.head(5))
 
 
print (table.query('Manager == ["Debra Henley"]'))
print (table.query('Status == ["pending", "won"]'))
 
 
# ********************************************
# -- 데이터 피벗(crosstab) : 별로!!!
# ********************************************
print("-" * 100 + "[4]") # ----- #
 
# -- crosstab([가로줄 덩어리], [세로줄덩어리])
print (pd.crosstab([data["Manager"], data["Rep"]], [data["Price"], data["Quantity"]]))
print (pd.crosstab([data["Manager"], data["Rep"]], [data["Price"], data["Quantity"]], margins=True))
print (pd.crosstab([data["Manager"], data["Rep"]], [data["Price"]]
            , values=data["Price"], aggfunc=[len, np.mean], margins=True))
 
print("-" * 100 + "[4.1]") # ----- #
 
# -- NULL 처리(flii_value없음, fillna사용)
print (pd.crosstab([data["Manager"], data["Rep"]], [data["Price"]]
            , values=data["Price"], aggfunc=[len, np.mean], margins=True).fillna(0))
 
 
# ********************************************
# -- 데이터 재형성(stack, unstack) : 거의 안 쓰임!!!
# ********************************************
print("-" * 100 + "[4.2]") # ----- #
 
# Row Multi-Index
row_idx_arr = list(zip(['r0', 'r0'], ['r-00', 'r-01']))
row_idx = pd.MultiIndex.from_tuples(row_idx_arr)
 
# Column Multi-Index
col_idx_arr = list(zip(['c0', 'c0', 'c1'], ['c-00', 'c-01', 'c-10']))
col_idx = pd.MultiIndex.from_tuples(col_idx_arr)
 
# Create the DataFrame
d = pd.DataFrame(np.arange(6).reshape(2, 3), index=row_idx, columns=col_idx)
d = d.applymap(lambda x: (x // 3, x % 3))
 
# Stack/Unstack
s = d.stack()
u = d.unstack()
 
print (d)
print (s)
print (u)
 
 
# ********************************************
# -- 그룹연산(groupby) : R에서도 많이 쓰임!!!
# ********************************************
print("-" * 100 + "[5]") # ----- #
 
import dateutil
 
data = pd.DataFrame.from_csv("./data/phone_data.csv")
print (data.head(3))
 
# -- 날짜 컬럼 자동 정규화 : datautil.parser.parse사용
data["date"] = data["date"].apply(dateutil.parser.parse, dayfirst=True)
print (data.head(3))
 
print("-" * 100 + "[5.1]") # ----- #
 
# -- 날짜 컬럼 자동 정규화 : pd.datetime.strptime 사용
data_path = "./data/phone_data.csv"
data = pd.DataFrame.from_csv(data_path)
dates = data["date"]
 
dateparse = lambda dates: [pd.datetime.strptime(d, "%y/%m/%d %H:%M") for d in dates]
 
data = pd.read_csv(data_path, parse_dates=["date"], date_parser=dateparse)
print (data.head(3))
 
print("-" * 100 + "[5.2]") # ----- #
 
# -- 전체 gropuby 연산
print (data["item"].count())
print (data["duration"].max())
print (data[data.item=="call"].max())
print (data["duration"][data.item=="call"].sum())
 
# -- groupby
data.groupby(["month"]).groups.keys()       # group key 보기
data.groupby(["month"]).groups.values()     # group value 보기
data.groupby(["month"]).groups["2014-11"]
 
for name, group in data.groupby(["month"]):
    print(str(name) + "  ||   " + str(group))
 
# -- groupby 연산
print (data.groupby(["month"]).first())   # group별 최초 값 호출
print (data.groupby(["month"]).mean())    # 숫자인 컬럼의 평균
print (data.groupby(["month"]).sum())
print (data.groupby(["month","item"]).sum())
 
# -- groupby 컬럼별 연산
print (data.groupby(["month"])["date"].count())
print (data[data["item"]=="call"].groupby("network")["duration"].sum())
 
# -- 여러 컬럼 groupby
print (data.groupby(["month", "item"]).count())
print (data.groupby(["month", "item"])["date"].count())
print (data.groupby(["month", "network_type"])["date"].count())
 
# -- 컬럼별 다른 연산(agg)
print (data.groupby(["month", "item"]).agg({"duration":sum, "network_type":"count", "date":"first"})) # 아래 pivot_table과 동일
print (pd.pivot_table(data, index=["month", "item"], aggfunc={"duration":sum, "network_type":"count", "date":"first"}))
