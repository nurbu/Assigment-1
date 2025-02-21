import kagglehub
import numpy as np
from Mnist_loader import MnistDataLoader
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB


MNIST = kagglehub.dataset_download("hojjatk/mnist-dataset")
loader = MnistDataLoader(MNIST)

(x_train,y_train),(x_test,y_test) = loader.data_loader()
train_img_flat = x_train.reshape(x_train.shape[0],-1)
test_img_flat = x_test.reshape(x_test.shape[0],-1)

#GuassianNB Model
Gmodel = GaussianNB()
Gmodel.fit(train_img_flat, y_train)


Gpredictions = Gmodel.predict(test_img_flat)


Gaccuracy = accuracy_score(y_test, Gpredictions)
print("GaussianNB Accuracy:", Gaccuracy)

#Mutlinomial Model
train_img_flat = (train_img_flat>128).astype(int)
test_img_flat = (test_img_flat>128).astype(int)
Mmodel = MultinomialNB()

Mmodel.fit(train_img_flat, y_train)

Mpredictions = Mmodel.predict(test_img_flat)

Maccuracy = accuracy_score(y_test,Mpredictions)
print("Multinomial Accuracy:", Maccuracy)

