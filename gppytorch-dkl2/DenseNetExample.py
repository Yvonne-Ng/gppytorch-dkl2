# copied directly from this:
#https://github.com/cornellius-gp/gpytorch/blob/master/examples/08_Deep_Kernel_Learning/Deep_Kernel_Learning_DenseNet_CIFAR_Tutorial.ipynb
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torch import nn
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import gpytorch
import math
import traceback
#import Thread

#Set up date augmentation
normalize = transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
aug_trans = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
common_trans = [transforms.ToTensor(), normalize]
train_compose = transforms.Compose(aug_trans + common_trans)
test_compose = transforms.Compose(common_trans)

# Create DataLoaders
dataset = 'cifar10'

if dataset == 'cifar10':
    d_func = dset.CIFAR10
    train_set = dset.CIFAR10('data', train=True, transform=train_compose, download=True)
    test_set = dset.CIFAR10('data', train=False, transform=test_compose)
    #train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)
    num_classes = 10
elif dataset == 'cifar100':
    d_func = dset.CIFAR100
    train_set = dset.CIFAR100('data', train=True, transform=train_compose, download=True)
    test_set = dset.CIFAR100('data', train=False, transform=test_compose)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=256, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    num_classes = 100
else:
    raise RuntimeError('dataset must be one of "cifar100" or "cifar10"')

# Creating the DenseNet Model
from densenet import DenseNet

class DenseNetFeatureExtractor(DenseNet):
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=self.avgpool_size).view(features.size(0), -1)
        return out

feature_extractor = DenseNetFeatureExtractor(block_config=(6, 6, 6), num_classes=num_classes).cpu()
num_features = feature_extractor.classifier.in_features

class GaussianProcessLayer(gpytorch.models.AdditiveGridInducingVariationalGP):
    def __init__(self, num_dim, grid_bounds=(-10., 10.), grid_size=64):
        super(GaussianProcessLayer, self).__init__(grid_size=grid_size, grid_bounds=[grid_bounds],
                                                   num_dim=num_dim, mixing_params=False, sum_output=False)
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x)
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        res = self.gp_layer(features)
        return res

model = DKLModel(feature_extractor, num_dim=num_features).cpu()
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, n_classes=num_classes).cpu()

n_epochs = 1
lr = 0.1
optimizer = SGD([
    {'params': model.feature_extractor.parameters()},
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()},
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)

def train(epoch):
    print("got here 1")
    model.train()
    likelihood.train()


    print("got here 2")
    mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

    train_loss = 0.
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cpu(), target.cpu()

        print("got here 2.1")
        optimizer.zero_grad()

        print("got here 2.2")
        output = model(data)

        print("got here 2.3")

        loss = -mll(output, target)

        print("got here 2.4")
        try:
            loss.backward()
        except:
            print("exception happened")
            traceback.print_exc()


        print("got here 2.5")
        optimizer.step()

        print("got here 2.6")
        if (batch_idx + 1) % 25 == 0:
            print('Train Epoch: %d [%03d/%03d], Loss: %.6f' % (epoch, batch_idx + 1, len(train_loader), loss.item()))

        print("got here 2.7")


    print("got here 3")

def test():
    from torch.optim import SGD, Adam
    from torch.optim.lr_scheduler import MultiStepLR
    import torch.nn.functional as F
    from torch import nn
    import torch
    model.eval()
    likelihood.eval()

    correct = 0
    for data, target in test_loader:
        data, target = data.cpu(), target.cpu()
        with torch.no_grad():
            output = likelihood(model(data))
            pred = output.probs.argmax(1)
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))

for epoch in range(1, n_epochs + 1):
    scheduler.step()
    with gpytorch.settings.use_toeplitz(False), gpytorch.settings.max_preconditioner_size(0):
        train(epoch)
        print("printing stack trace")
        #traceback.print_stack()
        #print(traceback.print_stack())
        #Thread.dumpStack()
        #print(Thread.dumpStack())
        test()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')


