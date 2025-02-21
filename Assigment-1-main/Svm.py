import kagglehub
import numpy as np
from Mnist_loader import MnistDataLoader
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

MNIST = kagglehub.dataset_download("hojjatk/mnist-dataset")
loader = MnistDataLoader(MNIST)

(x_train,y_train),(x_test,y_test) = loader.data_loader()
train_img_flat = x_train.reshape(x_train.shape[0],-1)
test_img_flat = x_test.reshape(x_test.shape[0],-1)
 

linear = SVC(kernel='linear')
linear.fit(train_img_flat,y_train)
linear_pred = linear.predict(test_img_flat)
laccuracy = accuracy_score(y_test,linear_pred)
print(laccuracy)

# rbf = SVC(Kernel="rbf")


