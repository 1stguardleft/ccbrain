'''
/*********************************************************************************************************
-- Title : [Py2.7] Pandas.DataFrame 사용법(생성, 추가, 삭제, 조회, 메타 등)
-- Reference : Python for Data Analysis
-- Key word : pandas dataframe 판다스 데이터프레임 
--출처: http://dbrang.tistory.com/994 [dBRang]
*********************************************************************************************************/
'''

# -*- coding: utf-8 -*-
 
from pandas import Series, DataFrame
import numpy as np
import pandas as pd
 
# ********************************************
# -- 데이터프레임 개체 생성(create)
# ********************************************
#    Table이나 Sheet형식이 데이터 저장 개체
#    indext와 여러 column으로 구성
 
# -- 기본 데이터프레임 생성
df = DataFrame([1000, 2000, 3000, 4000])
print(df)
 
df = DataFrame([1000, 2000, 3000, 4000], index=["i1", "i2", "i3", "i4"])
print(df)
 
df = DataFrame({"c1":[1000, 2000, 3000, 4000]}, index=["i1", "i2", "i3", "i4"])
print(df)
 
print("-" * 100 + "{[1]}") # ----- #
 
# -- 여러 컬럼 데이터프레임 생성
df2 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]})
print(df2)
 
df2 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
print(df2)
 
print("-" * 100 + "{[2]}") # ----- #
 
# -- 컬럼만 있는 데이터프레임 생성
df4 = DataFrame(columns=("lib","qt1","qt2"))
print(df4)
 
for i in range(5):
    df4.loc[i] = [(i+1)*(n+1) for n in range(3)]
print(df4)
 
print("-" * 100 + "{[3]}") # ----- #
 
# -- 난수를 이용한 데이터프레임 생성
df3 = DataFrame(np.random.randn(6,3))
print(df3)
 
print("-" * 100 + "{[4]}") # ----- #
 
# -- 리스트로 데이터프레임으로 생성
lst1 = [1,2,3,4]
df = DataFrame(lst1)
print(df)
 
lst2 = [[1,2,3,4,5], ["a", "b","c","d","e"]]
df = DataFrame(lst2)
print(df)
 
print("-" * 100 + "{[5]}") # ----- #
 
# -- 사전으로 데이터프레임으로 생성
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df0 = DataFrame(data, index=['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
print(df0)
 
print("-" * 100 + "{[5.5]}") # ----- #
 
# ********************************************
# -- 데이터프레임 개체 수정(alter)
# ********************************************
df = DataFrame([1000, 2000, 3000, 4000])
df2 = DataFrame({"x1":[1,2,3], "x2":[11,22,33], "x3":[111,222,333]})
df3 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
 
# -- 컬럼명 변경
df.columns = ["C1"]
print(df)
 
df2.columns = ["c1", "c2", "c3"]
print(df2)
 
df2.rename(columns={"c1":"CC1"}, inplace = True)  # 많이 쓰임!!!
print(df2)
 
print("-" * 100 + "{[6]}") # ----- #
 
df.columns.values[0] = 999       # 컬럼명이 숫자일 땐 숫자로..
print(df)
 
df3.columns.values[0] = "ttt"    # 컬럼명이 문자일 땐 문자로..
print(df3)
 
# -- 컬럼명 변경2
df8 = pd.DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
 
df8.columns.values[0] = "id_area8"
print(df8)
df8.rename(columns={df8.columns[0]:"id_area7"}, inplace = True)
print(df8)
 
# -- 컬럼 추가 : 값 지정
df["newC2"] = 5
print(df)
 
df["newC3"] = ["a","b","c","d"]
print(df)
 
# -- 컬럼 추가 : Boolean 조건 지정
df2["new9"] = df2["c3"] > 300
print(df2)
 
print("-" * 100 + "{[7]}") # ----- #
 
# -- 컬럼 추가 : 다른 column을 계산하여 생성
df2["newC9"] = df2["c2"] + df2["c3"]
print(df2)
 
print("-" * 100 + "{[8]}") # ----- #
 
# -- 컬럼 추가 : 시리즈로 추가
df9 = DataFrame([1000, 2000, 3000, 4000], index=["i1", "i2", "i3", "i4"])
print(df9)
 
add_srs = Series([1001, 2001, 3001, 4001], index = ["i1","i2","i3","i4"])
print(add_srs)
 
df9["c3"] = add_srs
print(df9)
 
add_srs = Series([1011, 3011, 4011], index = ["i3","i4","i5"]) # "i5"는 반영 안됨(outer join 방식)
df9["c4"] = add_srs
print(df9)
 
print("-" * 100 + "{[9]}") # ----- #
 
# -- 컬럼 추가 : 리스트로 추가(가장 편할 듯!!!)
add_lst = [1111,2222,3333,4444]
df9["c5"] = add_lst
 
print("-" * 100 + "{[10]}") # ----- #
 
# -- 컬럼 추가 : 빈 데이터프레임에 컬럼 추가(인덱스 포함)
df22 = DataFrame()
dd22 = df22.append({"c1":"", "c2":"", "c3":""}, ignore_index=True)
 
print(df22)
 
# -- 컬럼 추가 : 빈 데이터프레임에 컬럼 추가(인덱스 제외)
df33 = pd.DataFrame(columns=("c1", "c2", "c3"))

print(df33) 
print("-" * 100 + "{[10.3]}") # ----- #
 
# -- 컬럼 삭제
df2 = DataFrame({"x1":[1,2,3], "x2":[11,22,33], "x3":[111,222,333]})
df3 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
 
del df2["x2"]
print(df)
 
df2 = df2.drop("x1", 1)
print(df2)
 
df3 = df3.drop(["c2", "c3"], 1)
print(df3)
 
print("-" * 100 + "{[11]}") # ----- #
 
# -- 인덱스 컬럼명 변경      =====> 인덱스로 옮길까????
df2.index.name = "Count"
print(df2.index)
print(df2)
 
 
# ********************************************
# -- 데이터프레임 개체 속성 조회 및 변경(meta)
# ********************************************
df = DataFrame([1000, 2000, 3000, 4000], index=["i1","i2","i3","i4"])
df2 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
 
# -- 컬럼 이름 리턴
print(df2.columns.values)
 
# -- 인덱스 이름 리턴
print(df2.index)
 
# -- 기타 속성 리턴
print (df2.shape[0])  # 행 개수 : for문 range에 활용
print (df2.shape[1])  # 열 개수
print (df2.size)      # 행 * 열 개수
print (df2.ndim)      # 뭔지 모름?
print (df2.info())     # DF정보
print (df2.describe())  # DF 일괄 통계 정보 추출
 
print("-" * 100 + "{[12]}") # ----- #
 
# -- 개체 타입 확인
print(type(df.ix["i1"]))           # 결과 : pandas.core.series.Series
print(type(df.ix[1:3, 0:2]))       # 결과 : pandas.core.frame.DataFrame
 
print("-" * 100 + "{[12.5]}") # ----- #
 
 
# ********************************************
# -- 데이터프레임 데이터 조회(select)
# ********************************************
df = DataFrame([1000, 2000, 3000, 4000], index=["i1","i2","i3","i4"])
df2 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
 
# -- 원하는 컬럼만 출력
newColList = [0,1]
print (df2[newColList])
 
# -- 데이터 추출 : 컬럼단위
print (df2[["c1","c3"]])
print (df2[["c2"]])         # 인덱스와 컬럼명까지 출력
print (df2["c2"])           # 인덱스만 출력
 
print("-" * 100 + "{[13]}") # ----- #
 
# -- 데이터 추출 : 행단위
print (df2.iat[0,0])
print (df2.at["i1","c1"])
 
print (df2.loc["i2"])
print (df2.loc["i2"]["c1"])
print (df2.iloc[0,0:3])
 
print("-" * 100 + "{[14]}") # ----- #
 
# -- 데이터 추출 : 부분
print (df2.ix[0,0])
print (df2.ix[1:3, 0:2])
print (df2.ix["i1":"i2", "c2":"c3"])
 
print (df2.ix[:,["c2", "c3"]])
print (df2.ix[["i2","i3"], :])
 
print("-" * 100 + "{[15]}") # ----- #
 
# -- 데이터 추출 : 아래 3개 동일
print (df2[0:2])
print (df2.ix[0:2])
print (df2.ix[0:2, :])
 
print("-" * 100 + "{[16]}") # ----- #
 
# -- 데이터 추출 : 조건 처리
df2["c4"] = df2["c3"] > 300
print (df2)
 
print("-" * 100 + "{[17]}") # ----- #
 
 
# ********************************************
# -- 데이터프레임 데이터 변경(update, delete, insert)
# ********************************************
df2 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
 
# -- 컬럼값 전체 변경
df2["c3"] = df2["c1"] + df2["c2"]
print (df2)
 
df2["c3"] = 111
print (df2)
 
print("-" * 100 + "{[18]}") # ----- #
 
# -- 컬럼 추가 후 데이터 입력
df2["newC4"] = ["a","b","c"]
print (df2)
 
# -- 컬럼 데이터 수정
df2["newC4"] = [111,222,333]
print (df2)
 
df2.ix[1,2] = 2000
print (df)
 
print("-" * 100 + "{[19]}") # ----- #
 
# -- 행 추가
rows = [4, 44, 444, "d"]
df2.loc[len(df2)] = rows
print(df2)
 
df2 = df2.rename(index={3:"i4"})
print(df2)
 
# -- 행 추가 : 빈 데이터프레임에 행 추가(인덱스 포함)
df22 = DataFrame()
dd22 = df22.append({"c1":"", "c2":"", "c3":""}, ignore_index=True)
  
df22.ix[0, "c1"] = "111" 
df22.ix[0, "c2"] = "222"
df22.ix[0, "c3"] = "333"
 
print(df22)
 
# -- 행 추가 : 빈 데이터프레임에 행 추가(인덱스 제외)
df33 = pd.DataFrame(columns=("c1", "c2", "c3"))
 
df33.loc[0] = [ "111", "222", "333"]
 
print(df33)
 
 
# -- 행 삭제
df2 =  df2.drop("i2", axis = 0)
print(df2)
 
print("-" * 100 + "{[20]}") # ----- #
 
 
# ********************************************
# -- 데이터프레임 출력 변환
# ********************************************
print(df2.T)         # 피벗
print(df2.values)    # array로 받음
print(df2.shape)     # 행렬(Array) 형태 출력
 
print("-" * 100 + "{[21]}") # ----- #
 
 
# ********************************************
# -- 인덱싱(index, reindex)
# ********************************************
df = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
print(df)
 
df = df.reindex(["i1", "i2", "i3", "i4", "i5"])
print(df)
 
print("-" * 100 + "{[22]}") # ----- #
 
df = df.reindex(["i1", "i2", "i3", "i4", "i5"]).interpolate() # 보간법
print(df)
 
df["interPo"] = df["c1"].interpolate()
print(df)
 
print("-" * 100 + "{[23]}") # ----- #
 
# -- 중복 인덱싱
df = df.reindex(["i1", "i2", "i4", "i4", "i5"])
print(df)
 
print(df.index.is_unique)  # 인덱스가 unique이면 True 리턴
print(df.ix["i4"])
 
print("-" * 100 + "{[24]}") # ----- #
 
# -- 인덱스 컬럼명 변경
df7 = DataFrame({"c1":[1,2,3], "c2":[11,22,33], "c3":[111,222,333]}, index=["i1","i2","i3"])
 
df7.index.name = "id_area9"
 
print("-" * 100 + "{[24.5]}") # ----- #
 
# -- 기존 컬럼을 인덱스로 변경
df33 = DataFrame({"c1":[1,2], "c2":[11,22]}, index=["i1", "i2"])
print(df33)
 
df33.index.name = "idx"
print(df33)
 
df33 = df33.set_index("c1")
print(df33)
 
 
# ********************************************
# -- 정렬(sort_index, sort_values)
# ********************************************
df = DataFrame({"c2":[10,40,70], "c3":[50,20,20], "c1":[60,30,90]}, index=["i2","i1","i3"])
print(df)
 
# -- 정렬 : 인덱스 기준
print(df.sort_index())                           # 인덱스명으로 정렬
print (df.sort_index(axis=0))                     # 인덱스(?)로 정렬
print (df.sort_index(axis=1))                     # 컬럼명(?)으로 정렬
print (df.sort_index(axis=0,  ascending = False)) # descending
 
print("-" * 100 + "{[25]}") # ----- #
 
df2 = df.sort_index(axis = 0)
df3 = df2.sort_index(axis = 1)
print(df3)
 
print("-" * 100 + "{[26]}") # ----- #
 
# -- 정렬 : 컬럼 기준
print (df.sort_values(by = "c1"))
print (df.sort_values(by = "c1", ascending=False))
print (df.sort_values(by = ["c3","c2"]))
 
print("-" * 100 + "{[27]}") # ----- #
 
 
# ********************************************
# -- NULL 처리(NaN, NA, NULL)
# ********************************************
from numpy import nan as NA
df = DataFrame(np.random.randn(7,3))
print(df)
 
# -- 강제로 NA 추가
df.ix[2,1] = NA
df.ix[3,2] = NA
print(df)
 
print("-" * 100 + "{[28]}") # ----- #
 
 
# ********************************************
# -- 랭킹 및 집계처리(lambda, apply)
# ********************************************
data = {'name': ['Jason', 'Molly', 'Tina', 'Jake', 'Amy'],
        'year': [2012, 2012, 2013, 2014, 2014],
        'reports': [4, 24, 31, 2, 3],
        'coverage': [25, 94, 57, 62, 70]}
df = DataFrame(data, index=['Cochice', 'Pima', 'Santa Cruz', 'Maricopa', 'Yuma'])
print(df)
 
print("-" * 100 + "{[29]}") # ----- #
 
# -- 랭킹
# print df.rank(method="first")    # 에러나는데.. 무슨 numeric일때만?
print (df.rank(axis=0))             # 행단위로 순위
print (df.rank(axis=1))             # 컬럼단위로 순위
print (df.rank(ascending=False, method="max"))
 
print("-" * 100 + "{[30]}") # ----- #
 
#-- 랭킹 : 한 컬럼으로 순위 선정
df['coverageRanked'] = df['coverage'].rank(ascending=1)
print(df)
 
print("-" * 100 + "{[31]}") # ----- #
 
# -- lambda를 이용한 apply함수 사용
#    행이나 열에서의 max() - min() 활용
df = DataFrame({"c1":[20,30,90], "c2":[30,30,80], "c3":[100,40,50]}, index=["i1","i2","i3"])
print(df)
 
print (df.apply(lambda x:x.max() - x.min(), axis = 0))    # 행간 계산
 
df["newC4"]  = df.apply(lambda x:x.max() - x.min(), axis = 1)   # 컬럼간 계산
print(df)
 
print (df.apply(lambda x:x+1))    # 행/컬럼 모두