# MNIST-Alphabet-Superposition-CNN-DACON

## 구현 목적
본 코드는 데이콘에서 진행된 컴퓨터 비전 학습대회에 기반하여 작성되었다. 먼저 주어진 데이터셋은 아래 사진처럼 알파벳과 숫자가 중첩되어 있는 상태에서, 알파벳 부분은 연하게, 숫자부분은 겹쳐진 부분 이외의 데이터를 지우고, 겹쳐진 부분만 진하게 표시되어 있고, 이런 이미지에 따른 라벨로 알파벳과 숫자가 주어진다. 이렇게 주어진 2048개의 Train set으로 그 10배인 20480개의 Test set의 숫자들을 예측하는 것이 목적이다. 데이터셋은 다음 링크에서 받을 수 있다. 
https://dacon.io/competitions/official/235626/overview/
### Data sample

## 구현 내용

### Data Augmentation
먼저, 주어진 Train set의 수가 Test set에 비해 월등히 작아, Data Augmentation을 통해 Train set의 양을 늘릴 필요가 있었다. 따라서 Data_Augmentation.ipynb의 내용과 같이 torchvision 라이브러리에서 지원해주는 툴을 사용하여 Data Augmentation을 진행하였다. 코드는 다음과 같다.
```python
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from PIL import Image
from pandas import Series, DataFrame

data = pd.read_csv('train.csv', index_col=0)

data_ = np.array(data)
label=data.loc[:,"digit":"letter"].values
x_data=data.loc[:,"0":"783"].values

train_data_x, val_data_x, train_data_label, val_data_label = train_test_split(x_data, label, 
                                                                        test_size=0.2)
                                                                        
aug_f = transforms.Compose([transforms.Resize((28, 28)),
                            transforms.RandomAffine(30,translate=(0.1,0.1) ,
                                                    shear=10, scale=(0.7, 1.3))])
val_data_x=val_data_x.reshape(len(val_data_x),28,28)
val_data_x=val_data_x.astype(float)

train_data_x=train_data_x.reshape(len(train_data_x),28,28)
train_data_x=train_data_x.astype(float)

aug_img_val = []
aug_img_train = []
val_label=[]
train_label=[]

for i in range(len(val_data_x)):
    for _ in range(50):
        aug_img_val.append(np.array(aug_f(Image.fromarray(val_data_x[i]))))
        val_label.append(val_data_label[i])
aug_img_val=np.array(aug_img_val)
val_label=np.array(val_label)

for i in range(len(train_data_x)):
    for _ in range(50):
        aug_img_train.append(np.array(aug_f(Image.fromarray(train_data_x[i]))))
        train_label.append(train_data_label[i])
aug_img_train=np.array(aug_img_train)
train_label=np.array(train_label)

aug_img_val=aug_img_val.reshape(len(aug_img_val),-1)
aug_img_train=aug_img_train.reshape(len(aug_img_train),-1)

aug_data_val=np.hstack((val_label, aug_img_val))
aug_data_train=np.hstack((train_label, aug_img_train))

np.take(aug_data_val,np.random.permutation(aug_data_val.shape[0]),
        axis=0,out=aug_data_val)
np.take(aug_data_train,np.random.permutation(aug_data_train.shape[0]),
        axis=0,out=aug_data_train)
        
data_val=DataFrame(aug_data_val, columns=data.columns)
data_train=DataFrame(aug_data_train, columns=data.columns)

data_val.to_csv("val_augmented.csv")
data_train.to_csv("train_augmented.csv")
```
### RESNET test
기본적인 학습의 테스트를 위해 RESNET152 구조의 코드를 사용하여 학습을 진행하였다. 결과 Validation Set에 대한 정확도가 대략 86% 가량까지 수렴하는것을 확인 할 수 있었고, RESNET구조를 지속적으로 가공하여 Custom model을 제작하였다.

### Custom model
최종적으로 구현한 모델들 중 가장 높은 성능을 낸 model의 구조는 다음과 같다.

```python

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer_1 = nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,
                                 stride=1,padding=2)
        self.act_1 = nn.ReLU()
        self.conv2_bn1 = nn.BatchNorm2d(64)
        
        self.layer_2 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,
                                 stride=1,padding=2)
        self.act_2 = nn.ReLU()
        self.conv2_bn2 = nn.BatchNorm2d(64)
        
        self.layer_3 = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,
                                 stride=1,padding=2)
        self.act_3 = nn.ReLU()
        self.conv2_bn3 = nn.BatchNorm2d(64)
        
        self.max_1=nn.MaxPool2d(2,2)
        
        self.layer_4 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,
                                 stride=1,padding=2)
        self.act_4 = nn.ReLU()
        self.conv2_bn4 = nn.BatchNorm2d(128)

        self.layer_5 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,
                                 stride=1,padding=2)
        self.act_5 = nn.ReLU()
        self.conv2_bn5 = nn.BatchNorm2d(128)
        
        self.layer_6 = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=3,
                                 stride=1,padding=2)
        self.act_6 = nn.ReLU()
        self.conv2_bn6 = nn.BatchNorm2d(128)
        self.max_2=nn.MaxPool2d(2,2)
        
        self.layer_7 = nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,
                                 stride=1,padding=2)
        self.act_7 = nn.ReLU()
        self.conv2_bn7 = nn.BatchNorm2d(256)
        
        self.layer_8 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,
                                 stride=1,padding=2)
        self.act_8 = nn.ReLU()
        self.conv2_bn8 = nn.BatchNorm2d(256)
        
        self.layer_9 = nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,
                                 stride=1,padding=2)
        self.act_9 = nn.ReLU()
        self.conv2_bn9 = nn.BatchNorm2d(256)
        
        self.max_3=nn.MaxPool2d(2,2)
        
        self.layer_10 = nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,
                                 stride=1,padding=2)
        self.act_10 = nn.ReLU()
        self.conv2_bn10 = nn.BatchNorm2d(512)
        
        self.layer_11 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,
                                 stride=1,padding=2)
        self.act_11 = nn.ReLU()
        self.conv2_bn11 = nn.BatchNorm2d(512)
        
        self.layer_12 = nn.Conv2d(in_channels=512,out_channels=512,kernel_size=3,
                                 stride=1,padding=2)
        self.act_12 = nn.ReLU()
        self.conv2_bn12 = nn.BatchNorm2d(512)
        
        self.max_4=nn.MaxPool2d(2,2)
        
        self.layer_13 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=3,
                                 stride=1,padding=2)
        self.act_13 = nn.ReLU()
        self.conv2_bn13 = nn.BatchNorm2d(1024)
        
        self.layer_14 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,
                                 stride=1,padding=2)
        self.act_14 = nn.ReLU()
        self.conv2_bn14 = nn.BatchNorm2d(1024)
        
        self.max_5=nn.MaxPool2d(2,2)
        
        self.layer_15 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1,
                                 stride=1)
        self.act_15 = nn.ReLU()
        self.conv2_bn15 = nn.BatchNorm2d(1024)
        
        self.layer_16 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,
                                 stride=1,padding=2)
        self.act_16 = nn.ReLU()
        self.conv2_bn16 = nn.BatchNorm2d(1024)
        
        self.layer_17 = nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=1,
                                 stride=1)
        self.act_17 = nn.ReLU()
        self.conv2_bn17 = nn.BatchNorm2d(1024)
        
        

        
        self.fc_layer_1 = nn.Linear(49*1024,1000)
        self.act_18 = nn.ReLU()
        
        self.bnm1=nn.BatchNorm1d(1000)
        
        self.fc_layer_2 = nn.Linear(1000,1000)
        self.act_19 = nn.ReLU()
        
        self.bnm2=nn.BatchNorm1d(1000)
        
        self.fc_layer_3 = nn.Linear(1000,100)
        self.act_20 = nn.ReLU()
        
        self.bnm3=nn.BatchNorm1d(100)
        
        self.fc_layer_4 = nn.Linear(100,10)
        self.act_21 = nn.ReLU()
        
```
                                                                        
## 구현 결과
데이콘의 실제 Test Accuracy값은 88.6프로 정도의 결과로, 상위 24%정도에 머물렀지만, 본 경험을 토대로, 위성 사진 물체인식를 다루는 이후의 대회에도 다시 참여하여 조금 더 나아진 실력을 증명해 보려 한다.
