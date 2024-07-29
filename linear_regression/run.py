import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('./linear_regression/data_linear.csv').values
# N là số phần tử dataset
N = data.shape[0]
''' reshape biến mảng 1 chiều thành mảng 2 chiều 
1 là chỉ định rằng mảng con trong mảng lớn chỉ có 1 phần tử 
reshape(-1, 1) là cách thường dùng để chuyển đổi một mảng một chiều 
thành một mảng hai chiều với một cột'''
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)
plt.scatter(x, y)
plt.xlabel('mét vuông')
plt.ylabel('giá')
'''np.hstack() dùng để nối các mảng với nhau
    np.ones((N, 1)) là Tạo ra một mảng hai chiều có kích thước N×1, 
    trong đó tất cả các phần tử đều có giá trị là 1.
    mảng được tạo ra trong np.ones là mảng 2 chiều và mỗi mảng con 1 phần tử, 
    số 1 trong np.ones là số lượng phần tử trong mảng con
'''
x = np.hstack((np.ones((N, 1)), x))

w = np.array([0.,1.]).reshape(-1,1)

numOfIteration = 100
# mảng lưu cost sau mỗi lần gradient descent
cost = np.zeros((numOfIteration,1))
learning_rate = 0.00001
for i in range(1, numOfIteration):
    # tích vô hướng của w và x trừ y là residual đại diện cho phần dư, tức là sai số của model, y^ - y
    # r.shape = (30,1)
    r = np.dot(x, w) - y
    # tính cost, r được coi là phần dư tức là y^ - y
    cost[i] = 0.5*np.sum(r*r)
    # np.sum(r) tính toán toàn bộ độ sai lệch tên các điểm dữ liệu
    # ∂J/∂w[0] = ∑(r) = ∑(w0 + w1.xi -yi) vì f(w0,w1) = 1/2*(w0 + w1*xi - yi)^2 nên theo đạo hàm ta có
    w[0] -= learning_rate*np.sum(r)
    w[1] -= learning_rate*np.sum(np.multiply(r, x[:,1].reshape(-1,1)))
    
    print(cost[i])
predict = np.dot(x, w)
print(predict)
plt.plot((x[0][1], x[N-1][1]),(predict[0], predict[N-1]), 'r')
plt.show()
print(w)
x1 = 50
y1 = w[0] + w[1] * 50
print('Giá nhà cho 50m^2 là : ', y1)