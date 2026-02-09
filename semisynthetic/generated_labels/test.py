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


def test(M, prob_l):
    method_name = ['true_train_class', 'true_train_prob', 'MV', 'MV_probas', 'GLAD', 'GLAD_probas', 'DS', 'DS_probas']
    PATH = 'net_parameters/resnet18_finetune/'
    all_output = pd.read_csv('all_output.csv', index_col=0)

    for method in method_name:
        # torch.manual_seed(0)
        # np.random.seed(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = resnet18().to(device)
        model.load_state_dict(torch.load(PATH + method + '.pt'))
        model.eval()

        baseline_model = resnet18(pretrained=True).to(device)
        baseline_model.eval()

        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

        batch_size = 32

        test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

        correct = 0
        total = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_loader:
                images, labels = data[0].to(device), data[1].to(device)
                # calculate outputs by running images through the network
                outputs = model(images)
                baseline_outputs = baseline_model(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                _, baseline_predicted = torch.max(baseline_outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == baseline_predicted).sum().item()

        print(method+"-"+str(M)+"-"+str(prob_l)+": "+str(100 * correct / total)+"%")
        output = [M, prob_l, method, 100 * correct / total] + model.fc.weight.cpu().detach().numpy().flatten().tolist() + model.fc.bias.cpu().detach().numpy().tolist()
        output = pd.DataFrame(data=[output])
        if not all_output.empty:
            output.columns = all_output.columns
            all_output = pd.concat([all_output, output], axis=0, ignore_index=True)
        else:
            all_output = output
        print(all_output)

    all_output.to_csv('all_output.csv')
