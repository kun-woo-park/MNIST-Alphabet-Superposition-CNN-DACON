import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from pandas import DataFrame


def data_augmentation():
    # load data
    data = pd.read_csv('train.csv', index_col=0)
    label = data.loc[:, "digit":"letter"].values
    x_data = data.loc[:, "0":"783"].values

    # split data to train set and validation set
    train_data_x, val_data_x, train_data_label, val_data_label = train_test_split(x_data, label,
                                                                                  test_size=0.2)

    # define augmentation configure
    aug_f = transforms.Compose([transforms.Resize((28, 28)),
                                transforms.RandomAffine(30, translate=(0.1, 0.1),
                                                        shear=10, scale=(0.7, 1.3))])

    train_data_x = train_data_x.reshape(len(train_data_x), 28, 28)
    train_data_x = train_data_x.astype(float)

    val_data_x = val_data_x.reshape(len(val_data_x), 28, 28)
    val_data_x = val_data_x.astype(float)

    # list for augmentation data
    aug_img_val = []
    aug_img_train = []
    val_label = []
    train_label = []

    # data augmentation
    print("Start data augmentation")
    for i in range(len(val_data_x)):
        for _ in range(50):
            aug_img_val.append(np.array(aug_f(Image.fromarray(val_data_x[i]))))
            val_label.append(val_data_label[i])
    aug_img_val = np.array(aug_img_val)
    val_label = np.array(val_label)

    for i in range(len(train_data_x)):
        for _ in range(50):
            aug_img_train.append(
                np.array(aug_f(Image.fromarray(train_data_x[i]))))
            train_label.append(train_data_label[i])
    aug_img_train = np.array(aug_img_train)
    train_label = np.array(train_label)

    aug_img_val = aug_img_val.reshape(len(aug_img_val), -1)
    aug_img_train = aug_img_train.reshape(len(aug_img_train), -1)

    aug_data_val = np.hstack((val_label, aug_img_val))
    aug_data_train = np.hstack((train_label, aug_img_train))

    np.take(aug_data_val, np.random.permutation(aug_data_val.shape[0]),
            axis=0, out=aug_data_val)
    np.take(aug_data_train, np.random.permutation(aug_data_train.shape[0]),
            axis=0, out=aug_data_train)

    data_val = DataFrame(aug_data_val, columns=data.columns)
    data_train = DataFrame(aug_data_train, columns=data.columns)

    # save augmented data
    data_val.to_csv("val_augmented.csv")
    data_train.to_csv("train_augmented.csv")
