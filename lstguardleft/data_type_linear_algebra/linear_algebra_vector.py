# Linear Algebra 예제

import numpy as np
import numpy.linalg as linalg

''' 
벡터(Vector)
- 기하학적인 공간의 단위원소를 표현
- 각 성분의 데이터 타입이 동일
- 벡터는 행벡터와 열벡터로 구분
- python에서 vector는 numpy에서 제공하는 nparray 형태로 사용
- 벡터는 1차원 
'''

# 벡터의 표현
# a는 행벡터, 행벡터는 list 형태로 정의하고 이후 np.array 함수를 통해 nparray 데이터 구조로 변경

# 데이터형 : list
a = [1, 2, 3, 4]

# 데이터형 : nparray 
b = np.array(a)  

# 행벡터

print('=========================================')
print(type(a))
print(type(b))
print(b)
print(b.ndim) # 행벡터의 경우 (1, m) 이 아닌 (m, )로 표현된다 (1차원이므로)
print(b.shape) 
print('=========================================\n')

c = np.array([[1, 2, 3, 4]])
d = np.matrix(c)

print('=========================================')
print(type(c))
print(c)
print(c.ndim)
print(c.shape)

print(type(d))
print(d)
print(d.ndim)
print( d.shape)
print('=========================================\n')

# print(c == d)

# 열벡터

e = np.array([[1], [2], [3] ,[4]])
f = np.matrix(e)

print('=========================================')
print(type(e))
print(e)
print(e.ndim)
print(e.shape)

print(type(f))
print(f)
print(f.ndim)
print(f.shape)
print('=========================================\n')

# 벡터 연산
# 벡터 연산의 경우 
# 덧셈, 뺄셈, 스칼라곱, 성분곱 (Elementwise multiplication), 내적(inner product)

print('=========================================')

v = np.array([1, 2, 3, 4])
u = np.array([5, 6, 7, 8])

print(v + u)
print(v - u)

print(4*v)
print(np.multiply(5, v))

print(v*u)
print(np.multiply(v, u))

print(np.inner(v, u)) # 동일 인덱스상의 성분끼리의 곱, 위의 multiply와 동일 그렇다면 두 개를 의미상으로 정의한 까닭은 ????

print('=========================================\n')




