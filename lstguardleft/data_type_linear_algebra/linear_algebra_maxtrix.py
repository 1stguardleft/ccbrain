# Linear Algebra 예제

import numpy as np
import numpy.linalg as linalg

''' 
행렬 (Matrix)....
- 기하학적인 공간의 단위원소를 표현
- python에서 행렬은 numpy에서 제공하는 nparray 형태로 사용
- 벡터는 1차원, 행렬은 2차원 
'''

# 행렬의 표현
# list 형태로 정의하고 이후 np.array 함수를 통해 nparray 데이터 구조로 변경

# 데이터형 : list
a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]

# 데이터형 : nparray 
b = np.array(a)  

# 행벡터

print('=========================================')
print(type(a))
print(type(b))
print(b)
print(b.ndim) # 차원이 2차원
print(b.shape) 
print('=========================================\n')


c = [1, 2, 3, 4]
d = [5, 6, 7, 8]
e = [9, 10, 11, 12]

f = np.array([c, d, e])  # 행렬은 각성분이 배열 (벡터) 로 이루어진 배열임 !!!!!

print('=========================================')
print(type(f))
print(f)
print(f.ndim) 
print(f.shape) 
print('=========================================\n')

# 행렬연산

print(b + f)
print(b - f)

print(np.multiply(4, b))
print(np.multiply(b, f))

# 행렬과 벡터 연산의 가장 큰 차이점 : 행렬곱 (matrix multiplication)

x = np.random.normal(0, 1, (2, 3))
y = np.random.normal(0, 1, (3, 2))

print('=========================================')

print(np.matmul(x, y))
#np.matmul(y, x)
print('=========================================\n')

# 전치행렬 (Transpose)

print(x.shape)
print(np.transpose(x).shape)
print(x.T)

u = np.array([[1,2,3],[4,5,6]]) # (2,3) 행렬
v = np.array([[1,0,0],[0,1,0]]) # (2,3) 행렬
 
print('=========================================')

print(u*v)        # 성분곱 (2,3) 행렬
print(np.matmul(u.T, v)) # (3,2) mul (2, 3) = 행렬곱 (3,3) 
print(np.matmul(u, v.T)) # (2,3) mul (3, 2) = 행렬곱 (2,2) 
print('=========================================\n')

#  그 이외의 행렬들

z = np.random.normal(0, 1, (2, 2))

print(np.identity(3)) #단위행렬 (identity matrix)
print(linalg.inv(z))  #역행렬 (identity matrix), n * n
print(linalg.det(z))  #행렬식 (determinant)