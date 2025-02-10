import os
import struct
import numpy as np

class MnistDataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
        self.train_img_path = os.path.join(dataset_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        self.train_label_path = os.path.join(dataset_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        self.test_img_path = os.path.join(dataset_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        self.test_label_path = os.path.join(dataset_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    def read_img_label(self, img_path, label_path):
        with open(img_path, "rb") as f:
            magic_number, num_images, num_rows, num_columns = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num_images, num_rows, num_columns)
        
        with open(label_path, "rb") as f:
            magic_number, num_labels = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        
        return images, labels

    def data_loader(self):
        x_train, y_train = self.read_img_label(self.train_img_path, self.train_label_path)
        x_test, y_test = self.read_img_label(self.test_img_path, self.test_label_path)
        return (x_train, y_train), (x_test, y_test)
