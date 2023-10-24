import numpy as np
def sigmoid(x):
        return 1/(1+np.exp(-x))
class LogisticReg:
    def __init__(self, lr = 0.01, n_iter=1000):
        self.lr =lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
    
    

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights)+self.bias
            predictions = sigmoid(y_pred)

            dw = (1/n_samples)*np.dot(X.T, (predictions - y))
            db = (1/n_samples)*np.sum(predictions - y)
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
    
    def predict(self, X):
        y_pred = np.dot(X, self.weights)+self.bias
        predictions = sigmoid(y_pred)
        class_pred = [0 if y<=0.5 else 1 for y in predictions]
        return class_pred