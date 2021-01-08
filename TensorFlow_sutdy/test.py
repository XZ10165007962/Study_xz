# -*- coding: utf-8 -*-
# @Time : 2021/1/7 10:15
# @Author : BigZhuang
# @Site : 
# @File : test.py
# @Software: PyCharm
# @version:
# @alert:
import numpy as np

w = np.array([[1, 2], [3, 4]])
x = np.array([[1, 3], [2, 4]])
w_mat = np.mat([[1, 2], [3, 4]])
x_mat = np.mat([[1, 3], [2, 4]])
print(x)
print(w)
w_x_start = x * w
x_w_dot = np.dot(x, w)
x_w_matmul = np.matmul(x, w)
x_w_multiply = np.multiply(x, w)
print('-----多维运算------')
print(w_x_start)
print(x_w_dot)
print(x_w_matmul)
print(x_w_multiply)

array_x = np.array([1,2,3,4])
array_w = np.array([2,3,4,5])
print('----数组运算------')
w_x_start_arr = array_x * array_w
x_w_dot_arr = np.dot(array_x, array_w)
x_w_matmul_arr = np.matmul(array_x, array_w)
x_w_multiply_arr = np.multiply(array_x, array_w)
print(w_x_start_arr)
print(x_w_multiply_arr)
print(x_w_matmul_arr)
print(x_w_dot_arr)

print('----矩阵运算----')
print(x_mat * w_mat)
print(np.dot(x_mat, w_mat))
print(np.matmul(x_mat, w_mat))
print(np.multiply(x_mat, w_mat))
