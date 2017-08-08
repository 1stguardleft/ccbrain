'''
/*********************************************************************************************************
-- Title : [Py2.7] Pandas.Numpy 사용법
-- Reference : Python for Data Analysis
-- Key word : pandas numpy 판다스 
-- 출처: http://dbrang.tistory.com/992 [dBRang]
*********************************************************************************************************/
'''

# -*- coding: utf-8 -*-
 
# ********************************************
# -- numpy : 배열 구조 및 연산
# ********************************************
import numpy as np
 
# -- 난수를 통한 배열 생성
data = np.random.randn(2,3)
print(data)
 
print("-"*100) # ---------------------------
 
# -- 리스트를 통한 배열 생성
data1 = [6,7,5,8,0,1]
print(data1)
 
arr1 = np.array(data1)
print(arr1)
 
# -- 2차원 리스트를 통한 배열 생성
data2 = [[1,2,3,4],[5,6,7,8]]
arr2 = np.array(data2)
print(arr2)
 
print("-"*100) # ---------------------------
 
# -- array 구조 출력
print(arr2.shape)
 
# -- 0으로 채워진 배열 생성
print(np.zeros((2,3)))
 
# -- 순번으로 채워진 배열 생성
print(np.arange(15))
 
print("-"*100) # ---------------------------
 
# -- 다차원 배열 생성 및 계산
arr = np.array([[1,2,3],[4,5,6]])
print(arr)
print(arr+arr)
print(arr*arr)
 
print("-"*100) # ---------------------------
 
 
# ********************************************
# -- n차 배열
# ********************************************
 
# -- 다차원 배열 생성 및 조회
arr2d = np.array([[11,12,13],[14,15,16],[17,18,19]])
print(arr2d)
print(arr2d[1])
print(arr2d[1][2])
print(arr2d[:2, 1:])
 
print("-"*100) # ---------------------------
 
 
# ********************************************
# -- 불리언 색인(Boolean Index)
# ********************************************
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
data = np.random.randn(7,4)
 
print(names)
print(data)
 
# -- 조건에 맞는지 boolean 리턴
print(names == "Bob")
 
print("-"*100) # ---------------------------
 
# -- 두 배열을 조합한 조회
print(data[names=="Bob"])     # Bob이 있는 행
print(data[names=="Bob",0])   # Bob이 있는 행의 0번째 열
 
print("-"*100) # ---------------------------
 
# -- 배열의 OR 조건 Boolean 조회
mask = (names=="Bob") | (names=="Will")
print(mask)
 
# -- 조건에 따라 배열 값 변경 조회
data[data<0] = 0
print(data)
 
data[names!="Joe"] = 7
print(data)
 
print("-"*100) # ---------------------------
 
 
# ********************************************
# -- 피벗팅 및 배열 재조합
# ********************************************
 
# -- 피벗팅
arr = np.arange(15).reshape((3,5))
print(arr)
print(arr.T)
 
print("-"*100) # ---------------------------
 
# -- 재조합
aaa = np.arange(16)
print (aaa)
 
arr = np.arange(16).reshape(2,2,4)
print(arr)
 
# -- 피벗팅
arr2 = arr.swapaxes(1,2)
print (arr2)
 
 
# ********************************************
# -- 배열 조회 조건 및 집계
# ********************************************
 
arr = np.random.randn(4,4)
print(arr)
 
print(np.where(arr>0,1,-1))
print(arr)
 
print("-"*100) # ---------------------------
 
# -- where를 이용한 조건 조회
print(np.where((0<arr) & (arr<10), 2, -2))
print(np.where(np.logical_and(arr>0, arr<10), 3, -3))
print(np.where((0<arr) & (arr<10), arr, -4))
 
print("-"*100) # ---------------------------
 
# -- 배열 집계
print(arr.mean())
print(np.mean(arr))
print(arr.sum())
print(arr.cumsum())    # 이게 뭐더라?
print(arr.min())
print(arr.max())
 
print("-"*100) # ---------------------------
 
 
# ********************************************
# -- 정렬 밀 중복제거
# ********************************************
 
arr = np.random.randn(5,3)
print(arr)
 
# -- 정렬
arr.sort(1)     # 행 단위로 정렬
print(arr)
 
# -- 중복제거
names = np.array(["Bob", "Joe", "Will", "Bob", "Will", "Joe", "Joe"])
 
print(np.unique(names))        # 중복제거
print(len(np.unique(names)))   # 중복제거 개수
print(set(names))              # 뭐지?
print(np.unique(names)[1])     # 또 뭐더라?
 
print("-"*100) # ---------------------------
 
 
# ********************************************
# -- 선형대수(행렬 연산)
# ********************************************
x = np.array([[1,2,3],[4,5,6]])
y = np.array([[6,3],[-1,7],[8,9]])
print(x)
print(y)
 
# -- 행렬 곱하기?
print(x.dot(y))  # print(x*y)는 에러발생
 
print("-"*100) # ---------------------------
 
from numpy.linalg import inv, qr
x = np.random.randn(5,5)
 
mat = x.T.dot(x)
 
print(inv(mat))
print(mat.dot(inv(mat)))
 
 
# ********************************************
# -- 난수 생성
# ********************************************
samples = np.random.normal(size =(3,4))
print(samples)
 
samples = np.random.normal(size =(2,2))
print(samples)
 
data = np.random.randn(2,2)
print(data)
