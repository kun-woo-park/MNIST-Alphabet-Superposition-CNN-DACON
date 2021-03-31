from torch.utils.data import DataLoader
import torch.nn as nn
import os
import pandas as pd
from data_augmentation import data_augmentation
from deep_learning_modules import CustomDataset, Model, train_model

if __name__ == "__main__":
    # data augmentation
    if not (os.path.isfile('train_augmented.csv') or os.path.isfile('val_augmented.csv')):
        data_augmentation()
    # set number of usable gpus
    num_gpus = 4
    # set batch size
    batch_size = 256
    # total epoch for train
    total_epoch = 200
    # model version
    model_char = "minloss"
    # early stop patience
    patience = 5
    # early stop start epoch
    start_early_stop_check = 0
    # starting epoch for saving model
    saving_start_epoch = 10

    # load data
    train_data = pd.read_csv('train_augmented.csv', index_col=0)
    val_data = pd.read_csv('val_augmented.csv', index_col=0)
    x_data_train = train_data.loc[:, "0":"783"].values
    x_data_val = val_data.loc[:, "0":"783"].values
    y_data = train_data["digit"].values
    y_data_val = val_data["digit"].values
    # normalize by max value
    x_data_train = x_data_train/x_data_train.max()
    x_data_val = x_data_val/x_data_val.max()

    train_dataset = CustomDataset(x_data_train, y_data)
    train_loader = DataLoader(dataset=train_dataset, pin_memory=True,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=60, drop_last=True)
    val_dataset = CustomDataset(x_data_val, y_data_val)
    val_loader = DataLoader(dataset=val_dataset, pin_memory=True,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=60, drop_last=True)
    # define model
    model = nn.DataParallel(Model(batch_size, num_gpus))
    # train model
    train_model(model, train_loader, val_loader, batch_size, total_epoch, model_char,
                patience, start_early_stop_check, saving_start_epoch)
