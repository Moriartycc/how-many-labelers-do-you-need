import torch
import torchvision
import torchvision.transforms as transforms
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

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

model = vgg19_bn(pretrained=True)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616))])

batch_size = 100

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

true_count = np.load('cifar10h-counts.npy')
true_prob = np.load('cifar10h-probs.npy')
model.eval()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
with torch.no_grad():
    i = 0
    for data in testloader:
        i = i + 1
        print(i)
        images, labels = data
        output = feature_extractor(images)
        real_output = model(images)
        _, pred_class = torch.max(real_output, 1)
        pred_calib = torch.softmax(real_output, 1)
        if i == 1:
            predicted_class = pred_class
            predicted_calib = pred_calib
            true_class = labels
            outputs = output
        else:
            outputs = torch.cat((outputs, output))
            predicted_class = torch.cat((predicted_class, pred_class))
            predicted_calib = torch.cat((predicted_calib, pred_calib))
            true_class = torch.cat((true_class, labels))

# true_class = np.argmax(true_count, 1)
scipy.io.savemat('outputs.mat', mdict={'outputs': outputs.detach().numpy()})
scipy.io.savemat('pred.mat', mdict={'pred_class': predicted_class.detach().numpy(), 'pred_calib': predicted_calib.detach().numpy()})
scipy.io.savemat('true.mat', mdict={'true_count': true_count, 'true_class': true_class.detach().numpy(), 'true_calib': true_prob})

# correct_count = 0
# total_l1 = 0
# for i, j in zip(predicted_class, true_class):
#     if i == j:
#         correct_count += 1
# for i, j in zip(predicted_calib.detach().numpy(), true_calib):
#     total_l1 += 2 * np.abs(i-j)
# print("Classification accuracy is: {:.4f}".format(float(correct_count)/batch_size))
# print("Average L1 distance of calibration is: {:.4f}".format(float(total_l1)/batch_size))
