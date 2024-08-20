

# Thêm thư viện
import numpy as np
import pandas as pd
np.random.seed(5)
# Hàm sigmoid
def sigmoid(x):
    return 1/(1+np.exp(-x))
   
# Đạo hàm hàm sigmoid
def sigmoid_derivative(x):
    return x*(1-x)


# Lớp neural network
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
		# Mô hình layer ví dụ [2,2,1]
      self.layers = layers 
      
      # Hệ số learning rate
      self.alpha = alpha
		
      # Tham số W, b
      self.W = []
      self.b = []
      
      # Khởi tạo các tham số ở mỗi layer
      for i in range(0, len(layers)-1):
            
        #   vector w của layer i có shape là node của layer - 1 x node của layer i
            w_ = np.random.randn(layers[i], layers[i+1])
            # layer i+1 vì ở vị trí 0 là shape của input layer, có kích thước là node của layer i * 1
            b_ = np.zeros((layers[i+1], 1))
            #  chia cho số lượng node ở lớp i.giảm kích thước các trọng số ban đầu để tránh giá trị lớn ảnh hưởng đến quá trình học.
            self.W.append(w_/layers[i]) 
            self.b.append(b_)
    
	# Tóm tắt mô hình neural network
    def __repr__(self):
        return "Neural network [{}]".format("-".join(str(l) for l in self.layers))
    
	
	# Train mô hình với dữ liệu
    def fit_partial(self, x, y):
        A = [x]
        # quá trình feedforward
        out = A[-1]
        
        for i in range(0, len(self.layers) - 1):
            # z = x1*w1 + b1, i =0 : có out : 20X2(20 row,2 columns), self.W[i] : 2x2 (input layer: 2, hidden layer 1 : 2)
            out = sigmoid(np.dot(out, self.W[i]) + (self.b[i].T))
            A.append(out)
        # ma trận A sẽ lưu 3 array lần lượt là giá trị (x1,x2),(z1(1),z2(1)),(z1(2)) 
        # quá trình backpropagation
        y = y.reshape(-1, 1)
        # đạo hàm của loss function binary cross entropy
        dA = [-(y/A[-1] - (1-y)/(1-A[-1]))]
        dW = []
        db = []
        # A[-1] là phần tử cuối cùng trong ma trận A
        for i in reversed(range(0, len(self.layers)-1)):
            # công thức đạo hàm theo w,b https://nttuan8.com/bai-4-backpropagation/#Mo_hinh_tong_quat
            dw_ = np.dot((A[i]).T, dA[-1] * sigmoid_derivative(A[i+1]))
            db_ = (np.sum(dA[-1] * sigmoid_derivative(A[i+1]), 0)).reshape(-1,1)
            dA_ = np.dot(dA[-1] * sigmoid_derivative(A[i+1]), self.W[i].T)
            dW.append(dw_)
            db.append(db_)
            dA.append(dA_)
        
        # Đảo ngược dW, db
        '''
        các gradient dW và db được tính toán từ layer cuối cùng 
        về layer đầu tiên. Điều này có nghĩa là các gradient được tính toán 
        và lưu trữ theo thứ tự ngược lại (từ cuối về đầu) so với thứ tự của các layer trong mô hình.
        '''
        dW = dW[::-1]
        db = db[::-1]
        
		# Gradient descent
        for i in range(0, len(self.layers)-1):
            self.W[i] = self.W[i] - self.alpha * dW[i]
            self.b[i] = self.b[i] - self.alpha * db[i]
      
    def fit(self, X, y, epochs=20, verbose=10):
        for epoch in range(0, epochs):
            self.fit_partial(X, y)
            if epoch % verbose == 0:
                loss = self.calculate_loss(X, y)
                print("Epoch {}, loss {}".format(epoch, loss))
    
	# Dự đoán
    def predict(self, X):
        for i in range(0, len(self.layers) - 1):
            X = sigmoid(np.dot(X, self.W[i]) + (self.b[i].T))
        return X

	# Tính loss function
    def calculate_loss(self, X, y):
        y_predict = self.predict(X)
        #return np.sum((y_predict-y)**2)/2
        return -(np.sum(y*np.log(y_predict) + (1-y)*np.log(1-y_predict))) 
        
# Dataset bài 2
data = pd.read_csv('./neural_network/dataset.csv').values
N, d = data.shape
# giá trị ở cột cuối cùng là y nên phải cắt ra để train model
X = data[:, 0:d-1].reshape(-1, d-1)
y = data[:, 2].reshape(-1, 1)
p = NeuralNetwork([X.shape[1], 2, 1], 0.1)
p.fit(X, y, 1000, 100)

sample = X[0].reshape(1, -1)  # Reshape để đưa về định dạng đúng
prediction = p.predict(sample)
print("Dự đoán cho mẫu đầu tiên:")
print(prediction)