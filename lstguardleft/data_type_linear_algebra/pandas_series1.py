'''
/*********************************************************************************************************
-- Title : [Py2.7] Pandas.Series 사용법
-- Reference : Python for Data Analysis
-- Key word : 파이썬 pandas 판다스 series 시리즈
-- 출처: http://dbrang.tistory.com/990 [dBRang]
*********************************************************************************************************/
'''

# -*- coding: utf-8 -*-
 
from pandas import Series, DataFrame
import numpy as np
 
# *************************************
# -- 리스트와 시리즈의 차이
# *************************************
# -- 리스트는 value만으로 구성.
# -- 시리즈는 index + value로 구성.
#    index 미 지정시 0,1,2,3...으로 자동 생성.
 
sp_list = [100, 200, 300, 400, 500]
sp_series = Series([101, 202, 303, 404, 505])
 
print (sp_list)
print (sp_series)
 
print (sp_list[4])
print (sp_series[4])
 
print ("-" * 100) # --------------------
 
# -- 시리즈 생성
sp_series_withindex = Series([101, 202, 303, 404, 505], index = ["a","b","c","d","e"])
print (sp_series_withindex)
print (sp_series_withindex[1])
print (sp_series_withindex["b"])
 
print ("-" * 100) # --------------------
 
 
# *************************************
# -- 시리즈 조회(속성)
# *************************************
 
sp_list = [100, 200, 300, 400, 500]
sp_series = Series([101, 202, 303, 404, 505], index = ["a","b","c","d","e"])
 
# -- index 및 value 조회
print (sp_series.index)
print (sp_series.values)
 
for i in sp_series.index:
    print ("Index: " + i + "    Value: " + str(sp_series_withindex[i]))
 
print ("-" * 100) # --------------------
 
# -- index 유무 리턴
print ("c" in sp_series)
print ("f" in sp_series)
 
# -- 시리즈 계산
print (sp_series * 100)
 
# -- value 조건 추출
print (sp_series[sp_series.values > 300])  # 아래 명령어와 동일
print (sp_series[sp_series > 300])
 
print ("-" * 100) # --------------------
 
# -- 시리즈 이름 변경 및 속성 조회
sp_series.name = "stock_price"
print (sp_series)            # index + value 출력
print (sp_series.index)      # index 출력
print (sp_series.values)     # value 출력
print (sp_series.shape)      # (5L, ) : 행렬 구조 출력
print (sp_series.size)       # 5 : 사이즈? 출력
print (sp_series.ndim)       # 1 : 차원? 출력
 
print ("-" * 100) # --------------------
 
print (sp_series["c"])
print (sp_series[["a", "c"]])
 
print ("-" * 100) # --------------------
 
# -- 시리즈 value 개수 출력
print (sp_series.count())
print (len(sp_series))
 
print ("-" * 100) # --------------------
 
# *************************************
# -- index를 통한 시리즈 조회
#    (at/iat/loc/iloc/ix를 활용)
# *************************************
print (sp_series)
 
print ("-" * 100) # --------------------
 
# -- at:인덱스로 찾기(value only 추출)
print (sp_series.at["a"])
print (sp_series.iat[0])
print (sp_series.iat[2])
 
print ("-" * 100) # --------------------
 
# -- loc:인덱스 범위 찾기(index+value 추출), 많이 사용됨.
print (sp_series.loc["a":"c"])
print (sp_series.iloc[1:3])
 
print ("-" * 100) # --------------------
 
# -- 인덱스 찾기(지맘대로)
print (sp_series.ix[0])
print (sp_series.ix["b"])
 
print (sp_series.ix[[0,3]])
print (sp_series.ix[["a", "c"]])
 
print (sp_series.ix[0:3])
print (sp_series.ix["a":"c"])
 
print ("-" * 100) # --------------------
 
 
# *************************************
# -- 시리즈와 다른 개체와의 전환
# *************************************
 
# -- 시리즈에서 리스트로 추출(많이 쓰임)
search_index = [0, 3]
print ([sp_list[i] for i in search_index])
 
print ("-" * 100) # --------------------
 
# -- 사전을 시리즈로 변경
sdic = {"Ohio":1000, "Texas":2000, "Oregon":3000}
print (sdic)
 
srs3 = Series(sdic)
print (srs3)
 
# -- 리스트를 통해서 시리즈로 변경
states = ["California", "Ohio", "Oregon"]
srs4 = Series(sdic, index = states)
print (srs4)
 
# -- NULL 확인
print (srs4.isnull())
print (srs4.notnull())
 
print ("-" * 100) # --------------------
 
# -- 시리즈 연산(full join과 유사)
print (srs3 + srs4)
print (srs3 - srs4)
print (srs3 * srs4)
print (srs3 / srs4)
print (srs3 * 2 - srs4 * 3)
 
print ("-" * 100) # --------------------
 
# *************************************
# -- 계층적 시리즈
# *************************************
 
# -- 계층적 시리즈
hdata = Series(np.random.randn(10), index = [["a","a","a","b","b","b","c","c","d","d"],
                                             [1,2,3,1,2,3,1,2,2,3]])
print (hdata)
print (hdata.index)
print (hdata.values)
print (hdata["b"])
print (hdata["b", 2])
 
print ("-" * 100) # --------------------
 
print (hdata.shape)      # (10L, ) : 행렬 구조 출력
print (hdata.size)       # 10 : 사이즈? 출력
print (hdata.ndim)       # 1 : 차원? 출력
 
print ("-" * 100) # --------------------
 
 
# *************************************
# -- 시리즈 추가/삭제
# *************************************
 
# -- 시리즈 추가
add_data = Series([12345], index=["f"])
sp_series = sp_series.append(add_data)
print (sp_series)
 
# -- index 충돌시(False중복허용, True:에러발생)
add_data2 = Series([22222], index=["f"])
print (sp_series.append(add_data2, verify_integrity=False)) # True면 에러 발생.
 
print ("-" * 100) # --------------------
 
 
# *************************************
# -- 시리즈 reindex
# *************************************
 
# -- reindex
obj = Series([4.5, 7.2, -5.3], index = ["c", "b", "a"])
print (obj)
 
obj2 = obj.reindex(["a", "b", "c", "d", "e"])
print (obj2)
 
# -- fill_value
obj2 = obj.reindex(["a", "b", "c", "d", "e"], fill_value = 99.99)
print (obj2)
 
print ("-" * 100) # --------------------
 
# -- NaN 채우기 : method
#    ffill: 위에서 채우기, bfill: 아래에서 채우기
obj3 = obj.reindex(["a", "b", "d", "e", "c"], method="ffill") # 안되는 경우도 있더라.
print (obj3)
 
obj4 = obj.reindex(["a", "b", "d", "e", "c"], method="bfill")
print (obj4)
 
print ("-" * 100) # --------------------
 
# -- NaN 채우기 : interpolate(보간법)
obj5 = obj.reindex(["a", "b", "d", "e", "c"]).interpolate()
print (obj5)
