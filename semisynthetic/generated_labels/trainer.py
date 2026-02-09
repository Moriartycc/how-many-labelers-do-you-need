import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import torchvision.models as models
from cifar10_models.vgg import vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.resnet import resnet18, resnet34, resnet50
from typing import Any, Callable, Optional, Tuple
from cifar10_models.densenet import densenet121, densenet169, densenet161
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


class CIFAR10_new(torchvision.datasets.CIFAR10):
    def __init__(
            self,
            root: str,
            labels: pd.DataFrame,
            train: bool = True,
            download: bool = False,
            transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, train=train, download=download, transform=transform)
        self.transform = transform
        self.targets = torch.from_numpy(np.asarray(labels.values[:, 1:].tolist()))
        if self.targets.size(dim=1) == 1:
            self.targets = torch.flatten(self.targets).to(torch.int64)
        # print(self.targets)


def training(epoch_size=20, method_name=['MV', 'MV_probas', 'GLAD', 'GLAD_probas', 'DS', 'DS_probas']):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

    # method_name = ['true_train_prob', 'true_train_class', 'MV_probas', 'MV']
    # method_name = ['true_train_prob', 'true_train_class', 'MV', 'MV_probas', 'GLAD', 'GLAD_probas', 'DS', 'DS_probas']
    # method_name = ['MV', 'MV_probas', 'GLAD', 'GLAD_probas', 'DS', 'DS_probas']
    alpha_set = pd.read_csv('alpha_set.csv')

    for method in method_name:
        # torch.manual_seed(0)
        # np.random.seed(0)
        model = resnet18(pretrained=True).to(device)
        baseline_model = resnet18(pretrained=True).to(device)
        model.eval()
        baseline_model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, 10).to(device)
        model.fc.weight = nn.Parameter(alpha_set.values[-1, 1] * baseline_model.fc.weight).to(device)
        model.fc.bias = nn.Parameter(alpha_set.values[-1, 1] * baseline_model.fc.bias).to(device)

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
        # print(model)
        # print(device)

        batch_size = 32
        labels = pd.read_csv(method + '.csv')

        # train_set = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
        train_set = CIFAR10_new(root='../data', labels=labels, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)




        criterion = nn.CrossEntropyLoss()
        last_layer_parameters = [param for param in model.parameters() if param.requires_grad]
        optimizer = optim.SGD(last_layer_parameters, lr=1e-3, momentum=0.9)

        for epoch in range(epoch_size):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)
                # print(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                # outputs2 = baseline_model(inputs)
                # print(labels)
                # print(torch.softmax(outputs, 1))
                # print(torch.softmax(alpha_set.values[-1, 1] * outputs2, 1))
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 5 == 0:  # print every 5 mini-batches
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 5:.3f}')
                    running_loss = 0.0

        # print('Finished Training')

        PATH = 'net_parameters/resnet18_finetune/' + method + '.pt'
        torch.save(model.state_dict(), PATH)