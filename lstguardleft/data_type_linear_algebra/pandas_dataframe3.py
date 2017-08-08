'''
/*********************************************************************************************************
-- Title : [Py2.7] Pandas.DataFrame 조작 - 병합 및 조인(concat, append, join, merge)
-- Reference : Python for Data Analysis
-- Key word : pandas dataframe concat append join merger 데이터프레임 concatenate
-- 출처: http://dbrang.tistory.com/1001 [dBRang]
*********************************************************************************************************/
'''

# -*- coding: utf-8 -*-
 
import os, sys
import pandas as pd
 
 
# ********************************************
# -- concat, append를 활용한 DF 합치기
# ********************************************
# -- concat : 행, 컬럼 모두 가능
# -- append : 행으로 붙일 때 사용
print("-" * 100 + "{[1]}") # ----- #
 
 
# -- 행으로 붙이기
df1 = pd.DataFrame({"A":["a0", "a1","a2", "a3"],
                    "B":["b0", "b1","b2", "b3"],
                    "DD":["c0", "c1","c2", "c3"]},
                    index =[0,1,2,3]
                    )
print(df1)
 
df2 = pd.DataFrame({"A": ["a00", "a11", "a22"],
                    "B": ["b00", "b11", "b22"],
                    "CC": ["c00", "c11", "c22"]},
                   index=[2, 3, 4]
                   )
print (df2)
 
print("-" * 100 + "{[1.1]}") # ----- #
 
print (df1.append(df2))
print (pd.concat([df1, df2]))
 
print("-" * 100 + "{[1.2]}") # ----- #
 
# -- 인덱스 다시 만들기
print (pd.concat([df1, df2], ignore_index=True))
 
print("-" * 100 + "{[1.3]}") # ----- #
 
# -- 컬럼으로 붙이기
df1 = pd.DataFrame({"A": ["a0", "a1", "a2", "a3"],
                    "B": ["b0", "b1", "b2", "b3"],
                    "DD": ["c0", "c1", "c2", "c3"]},
                   index=[0, 1, 2, 3]
                   )
print (df1)
 
df2 = pd.DataFrame({"A": ["a00", "a11", "a22"],
                    "B": ["b00", "b11", "b22"],
                    "CC": ["c00", "c11", "c22"]},
                   index=[2, 3, 4]
                   )
print (df2)
 
print("-" * 100 + "{[1.4]}") # ----- #
 
print (pd.concat([df1, df2], axis=1)) # 컬럼으로 붙이기
print (pd.concat([df1, df2], axis=0))  # 행으로 붙이기
 
print("-" * 100 + "{[1.5]}") # ----- #
 
# -- 원하는 정보만 추출
print (pd.concat([df1, df2], axis=1).loc[:,["A","B"]])
print (pd.concat([df1, df2], axis=1).loc[1:3,["A","B"]])
print (pd.concat([df1, df2], axis=1).iloc[:,[0,1,2,3]])
 
print("-" * 100 + "{[1.6]}") # ----- #
 
# -- join 결과 가져오기
print (pd.concat([df1, df2], axis=1, join='inner'))
print (pd.concat([df1, df2], axis=1).dropna())      # dropna() : NULL없는 행만 가져오기
 
print("-" * 100 + "{[1.7]}") # ----- #
 
# -- left/right outer 결과 가져오기
pd.concat([df1, df2], axis=1, join="outer", join_axes=[df2.index])    # right join 결과
pd.concat([df1, df2], axis=1, join="outer", join_axes=[df1.index])    # left join 결과
 
 
# ********************************************
# -- join, merge를 활용한 DF 연결
#    (주로 merge가 SQL JOIN 수행)
# ********************************************
print("-" * 100 + "{[2]}") # ----- #
 
df1 = pd.DataFrame({"name": ["권혁", "이범호", "윌", "박병호", "양준혁"],
                    "teamno": [1001, 1002, 1002, 1003, None]},
                   index=["p01", "p02", "p03", "p04", "p05"]
                   )
print (df1)
 
df2 = pd.DataFrame({"teamid": [1001, 1002, 1003, 1004, 1005],
                    "teamname": ["엘지", "한화", "기아", "넥센", "삼성"],
                    "temehome": ["서울", "대전", "광주", "고척", "부산"]},
                   index=[0, 1, 2, 3, 4]
                   )
print (df2)
 
print("-" * 100 + "{[2.1]}") # ----- #
 
# -- inner join
print (pd.merge(df1, df2, how="inner", left_on="teamno", right_on="teamid"))
 
print("-" * 100 + "{[2.2]}") # ----- #
 
# -- left/right outer join
print (pd.merge(df1, df2, how="left", left_on="teamno", right_on="teamid"))
print (pd.merge(df1, df2, how="right", left_on="teamno", right_on="teamid"))
 
print("-" * 100 + "{[2.3]}") # ----- #
 
# -- left/right exculsive join
temp = pd.merge(df1, df2, how="left", left_on="teamno", right_on="teamid")
print (temp[temp["teamno"].isnull()])
 
print("-" * 100 + "{[2.4]}") # ----- #
 
# -- full join
print (pd.merge(df1, df2, how="outer", left_on="teamno", right_on="teamid"))
 
print("-" * 100 + "{[2.5]}") # ----- #
 
# -- full exclusive join
temp = pd.merge(df1, df2, how="outer", left_on="teamno", right_on="teamid")
print (temp[temp["teamno"].isnull() | temp["teamid"].isnull()])
 
print("-" * 100 + "{[2.6]}") # ----- #
 
# -- index 커럼을 가지고 join하기
 
df1 = pd.DataFrame({"name": ["권혁", "이범호", "윌", "박병호", "양준혁"],
                    "teamno": [1001, 1002, 1002, 1003, None]},
                   index=["p01", "p02", "p03", "p04", "p05"]
                   )
df1.index.name = "payerid"
print (df1)
 
df2 = pd.DataFrame({"teamname": ["엘지", "한화", "기아", "넥센", "삼성"],
                    "temehome": ["서울", "대전", "광주", "고척", "부산"]},
                   index=[1001, 1002, 1003, 1004, 1005]
                   )
print (df2)
 
print("-" * 100 + "{[2.7]}") # ----- #
 
# -- 인덱스를 가지고 조인
print(pd.merge(df1, df2, how="left", left_on="teamno", right_index=True))