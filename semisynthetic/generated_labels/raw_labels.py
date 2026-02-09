import torch
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import torchvision.models as models
from cifar10_models.vgg import vgg13_bn, vgg16_bn, vgg19_bn
from cifar10_models.resnet import resnet18, resnet34, resnet50
from cifar10_models.densenet import densenet121, densenet169, densenet161
from cifar10_models.mobilenetv2 import mobilenet_v2
from cifar10_models.googlenet import googlenet
from cifar10_models.inception import inception_v3
import scipy.io
import matplotlib.pyplot as plt
import numpy as np


def median_prob(scores):
    prob_max, _ = torch.max(torch.softmax(scores, 1), 1)
    return torch.median(prob_max)


def find_alpha(predicted_output, threshold, tol=1e-9):
    l = 0
    u = 10
    while u - l > tol:
        m = (l + u) / 2
        if median_prob(m * predicted_output) > threshold:
            u = m
        else:
            l = m
    return m


def generate_raw_labels(M, prob_l, prob_multiplier=1+1e-8):
    # torch.manual_seed(0)
    # np.random.seed(0)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet18(pretrained=True).to(device)
    # print(model)

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

    batch_size = 32

    train_set = torchvision.datasets.CIFAR10(root='../data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False, num_workers=0)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model.eval()
    feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
    with torch.no_grad():
        for i, data in enumerate(train_loader, 0):
            # print(i)
            images, labels = data[0].to(device), data[1].to(device)
            output = feature_extractor(images)
            pred_output = model(images)
            _, pred_class = torch.max(pred_output, 1)
            pred_calib = torch.softmax(pred_output, 1)
            if i == 0:
                predicted_class = pred_class
                predicted_calib = pred_calib
                predicted_output = pred_output
                true_class = labels
                outputs = output
            else:
                outputs = torch.cat((outputs, output))
                predicted_class = torch.cat((predicted_class, pred_class))
                predicted_calib = torch.cat((predicted_calib, pred_calib))
                predicted_output = torch.cat((predicted_output, pred_output))
                true_class = torch.cat((true_class, labels))

    # Find alpha region
    alpha_l = find_alpha(predicted_output, prob_l)
    alpha_u = alpha_l * prob_multiplier
    alpha_set = np.arange(alpha_l, alpha_u, (alpha_u - alpha_l) / M)

    # Generate fake labels from weak learners
    votes = torch.zeros(predicted_output.size(), dtype=int)
    votes_by_tasks = pd.DataFrame(0, index=np.arange(M * predicted_output.size()[0]),
                                  columns=['task', 'worker', 'label'])
    voter_count = 0
    total_count = 0
    for alpha in alpha_set:
        voter_count += 1
        # print(voter_count)
        # print(alpha)
        alpha_prob = torch.softmax(alpha * predicted_output, dim=1).cpu()
        # print(alpha_prob)
        for i in range(0, alpha_prob.size()[0]):
            task_vote = np.random.multinomial(1, alpha_prob[i, :])
            task_label = np.argmax(task_vote)
            votes[i, :] += task_vote
            single_task = [i, voter_count, task_label]
            votes_by_tasks.loc[total_count] = single_task
            total_count += 1

    vote = {'raw_votes': pd.DataFrame(votes.numpy()), 'task_votes': votes_by_tasks}
    vote['raw_votes'].to_csv("raw_votes.csv")
    vote['task_votes'].to_csv("task_votes.csv")
    torch.save(vote, 'raw_votes.pt')
    torch.save(alpha_set, 'alpha_set.pt')
    pd.DataFrame(alpha_set).to_csv("alpha_set.csv")
    df_predicted_output = pd.DataFrame(
        torch.reshape(predicted_output.cpu(), (predicted_output.size(dim=0), predicted_output.size(dim=1))).numpy())
    torch.save(df_predicted_output, 'train_last_layer.pt')
    df_predicted_output.to_csv("train_last_layer.csv")
    torch.save(true_class.cpu(), 'true_train_class.pt')
    pd.DataFrame(true_class.cpu()).to_csv("true_train_class.csv")
    pd.DataFrame(alpha_prob).to_csv("true_train_prob.csv")
    df_output = pd.DataFrame(torch.reshape(outputs.cpu(), (outputs.size(dim=0), outputs.size(dim=1))).numpy())
    df_output.to_csv("train_features")

    # Generate test features
    test_set = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        i = 0
        for data in test_loader:
            i = i + 1
            # print(i)
            images, labels = data[0].to(device), data[1].to(device)
            output = feature_extractor(images)
            pred_output = model(images)
            _, pred_class = torch.max(pred_output, 1)
            pred_calib = torch.softmax(pred_output, 1)
            if i == 1:
                predicted_class = pred_class
                predicted_calib = pred_calib
                predicted_output = pred_output
                true_class = labels
                outputs = output
            else:
                outputs = torch.cat((outputs, output))
                predicted_class = torch.cat((predicted_class, pred_class))
                predicted_calib = torch.cat((predicted_calib, pred_calib))
                predicted_output = torch.cat((predicted_output, pred_output))
                true_class = torch.cat((true_class, labels))
    df_predicted_output = pd.DataFrame(
        torch.reshape(predicted_output.cpu(), (predicted_output.size(dim=0), predicted_output.size(dim=1))).numpy())
    torch.save(df_predicted_output, 'test_last_layer.pt')
    df_predicted_output.to_csv("test_last_layer.csv")
    torch.save(true_class.cpu(), 'true_class.pt')
    pd.DataFrame(true_class.cpu()).to_csv("true_class.csv")
    df_output = pd.DataFrame(torch.reshape(outputs.cpu(), (outputs.size(dim=0), outputs.size(dim=1))).numpy())
    df_output.to_csv("test_features")
