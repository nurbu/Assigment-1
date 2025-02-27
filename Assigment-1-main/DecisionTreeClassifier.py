import kagglehub
import matplotlib.pyplot as plt
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from Mnist_loader import MnistDataLoader

MNIST = kagglehub.dataset_download("hojjatk/mnist-dataset")

loader = MnistDataLoader(MNIST)

(x_train,y_train),(x_test,y_test) = loader.data_loader()
train_img_flat = x_train.reshape(x_train.shape[0],-1)
test_img_flat = x_test.reshape(x_test.shape[0],-1)

model = DecisionTreeClassifier()
model.fit(train_img_flat,y_train)
predicitons = model.predict(test_img_flat)

accuracy = accuracy_score(y_test,predicitons)
print(accuracy)
