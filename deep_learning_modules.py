import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


class CustomDataset(Dataset):                   # custom dataset
    def __init__(self, x_dat, y_dat):
        x = x_dat
        y = y_dat
        self.len = x.shape[0]
        y = y.astype('int')
        x = x.astype('float32')
        self.x_data = torch.tensor(x)
        self.y_data = torch.tensor(y)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class Model(torch.nn.Module):                   # custom model
    def __init__(self, batch_size, num_gpus):
        super(Model, self).__init__()
        self.batch_size = batch_size
        self.num_gpus = num_gpus
        self.layer_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3,
                                 stride=1, padding=2)
        self.act_1 = nn.ReLU()
        self.conv2_bn1 = nn.BatchNorm2d(64)

        self.layer_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                 stride=1, padding=2)
        self.act_2 = nn.ReLU()
        self.conv2_bn2 = nn.BatchNorm2d(64)

        self.layer_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3,
                                 stride=1, padding=2)
        self.act_3 = nn.ReLU()
        self.conv2_bn3 = nn.BatchNorm2d(64)

        self.max_1 = nn.MaxPool2d(2, 2)

        self.layer_4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3,
                                 stride=1, padding=2)
        self.act_4 = nn.ReLU()
        self.conv2_bn4 = nn.BatchNorm2d(128)

        self.layer_5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                 stride=1, padding=2)
        self.act_5 = nn.ReLU()
        self.conv2_bn5 = nn.BatchNorm2d(128)

        self.layer_6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,
                                 stride=1, padding=2)
        self.act_6 = nn.ReLU()
        self.conv2_bn6 = nn.BatchNorm2d(128)
        self.max_2 = nn.MaxPool2d(2, 2)

        self.layer_7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3,
                                 stride=1, padding=2)
        self.act_7 = nn.ReLU()
        self.conv2_bn7 = nn.BatchNorm2d(256)

        self.layer_8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                 stride=1, padding=2)
        self.act_8 = nn.ReLU()
        self.conv2_bn8 = nn.BatchNorm2d(256)

        self.layer_9 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3,
                                 stride=1, padding=2)
        self.act_9 = nn.ReLU()
        self.conv2_bn9 = nn.BatchNorm2d(256)

        self.max_3 = nn.MaxPool2d(2, 2)

        self.layer_10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3,
                                  stride=1, padding=2)
        self.act_10 = nn.ReLU()
        self.conv2_bn10 = nn.BatchNorm2d(512)

        self.layer_11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                  stride=1, padding=2)
        self.act_11 = nn.ReLU()
        self.conv2_bn11 = nn.BatchNorm2d(512)

        self.layer_12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3,
                                  stride=1, padding=2)
        self.act_12 = nn.ReLU()
        self.conv2_bn12 = nn.BatchNorm2d(512)

        self.max_4 = nn.MaxPool2d(2, 2)

        self.layer_13 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3,
                                  stride=1, padding=2)
        self.act_13 = nn.ReLU()
        self.conv2_bn13 = nn.BatchNorm2d(1024)

        self.layer_14 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3,
                                  stride=1, padding=2)
        self.act_14 = nn.ReLU()
        self.conv2_bn14 = nn.BatchNorm2d(1024)

        self.max_5 = nn.MaxPool2d(2, 2)

        self.layer_15 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1,
                                  stride=1)
        self.act_15 = nn.ReLU()
        self.conv2_bn15 = nn.BatchNorm2d(1024)

        self.layer_16 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3,
                                  stride=1, padding=2)
        self.act_16 = nn.ReLU()
        self.conv2_bn16 = nn.BatchNorm2d(1024)

        self.layer_17 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1,
                                  stride=1)
        self.act_17 = nn.ReLU()
        self.conv2_bn17 = nn.BatchNorm2d(1024)

        self.fc_layer_1 = nn.Linear(49*1024, 1000)
        self.act_18 = nn.ReLU()

        self.bnm1 = nn.BatchNorm1d(1000)

        self.fc_layer_2 = nn.Linear(1000, 1000)
        self.act_19 = nn.ReLU()

        self.bnm2 = nn.BatchNorm1d(1000)

        self.fc_layer_3 = nn.Linear(1000, 100)
        self.act_20 = nn.ReLU()

        self.bnm3 = nn.BatchNorm1d(100)

        self.fc_layer_4 = nn.Linear(100, 10)
        self.act_21 = nn.ReLU()

    def forward(self, x):
        x = x.view(self.batch_size//self.num_gpus, 1, 28, 28)
        out = self.layer_1(x)
        out = self.act_1(out)
        for module in list(self.modules())[2:-11]:
            out = module(out)
        out = out.view(self.batch_size//self.num_gpus, -1)
        for module in list(self.modules())[-11:]:
            out = module(out)
        return out


def train_model(model, train_loader, val_loader, batch_size, total_epoch, model_char,
                patience, start_early_stop_check, saving_start_epoch):
    if torch.cuda.is_available():
        model.cuda()
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = torch.optim.Adam(model.parameters())

    # set loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.001)
    model_name = ""
    trn_loss_list = []
    val_loss_list = []

    for epoch in range(total_epoch):
        trn_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
            # grad init
            optimizer.zero_grad()
            # forward propagation
            output = model(inputs)
            # calculate loss
            loss = criterion(output, labels)
            # back propagation
            loss.backward()
            # weight update
            optimizer.step()

            # trn_loss summary
            trn_loss += loss.item()
        # validation
        with torch.no_grad():
            val_loss = 0.0
            cor_match = 0
            for j, val in enumerate(val_loader):
                val_x, val_label = val
                if torch.cuda.is_available():
                    val_x = val_x.cuda()
                    val_label = val_label.cuda()
                val_output = model(val_x)
                v_loss = criterion(val_output, val_label)
                val_loss += v_loss
                _, predicted = torch.max(val_output, 1)
                cor_match += np.count_nonzero(predicted.cpu().detach()
                                              == val_label.cpu().detach())

        trn_loss_list.append(trn_loss/len(train_loader))
        val_loss_list.append(val_loss/len(val_loader))
        val_acc = cor_match/(len(val_loader)*batch_size)
        now = time.localtime()
        print("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon,
              now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

        print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | val accuracy: {:.4f}% \n".format(
            epoch+1, total_epoch, trn_loss /
            len(train_loader), val_loss / len(val_loader), val_acc*100
        ))
        # early stop
        if epoch+1 > 2:
            if val_loss_list[-1] > val_loss_list[-2]:
                start_early_stop_check = 1
        else:
            val_loss_min = val_loss_list[-1]

        if start_early_stop_check:
            early_stop_temp = val_loss_list[-patience:]
            if all(early_stop_temp[i] < early_stop_temp[i+1] for i in range(len(early_stop_temp)-1)):
                print("Early stop!")
                break
        # save the minimum loss model
        if epoch+1 > saving_start_epoch:
            if val_loss_list[-1] < val_loss_min:
                if os.path.isfile(model_name):
                    os.remove(model_name)
                val_loss_min = val_loss_list[-1]
                model_name = "Custom_model_"+model_char + \
                    "_{:.3f}".format(val_loss_min)
                torch.save(model, model_name)
                print("Model replaced and saved as ", model_name)

    torch.save(model, "Custom_model_fin")
    print("model saved complete")
