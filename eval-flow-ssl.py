import myexman
import torch
import utils
import datautils
import os
from logger import Logger
import time
import numpy as np
from models import flows, distributions
import warnings
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
import timm
from multivae.data.datasets import MnistSvhn
from torch.utils.data import DataLoader
from multivae.data.utils import set_inputs_to_device
from sklearn.decomposition import PCA


def get_metrics(model, loader):
    logp, acc = [], []
    for x, y in loader:
        x = x.to(device)
        log_det, z = model.flow(x)
        log_prior_full = model.prior.log_prob_full(z)
        pred = torch.softmax(log_prior_full, dim=1).argmax(1)
        logp.append(utils.tonp(log_det + model.prior.log_prob(z)))
        acc.append(utils.tonp(pred) == utils.tonp(y))
    return np.mean(np.concatenate(logp)), np.mean(np.concatenate(acc))


parser = myexman.ExParser(file=os.path.basename(__file__))
parser.add_argument('--name', default='')
parser.add_argument('--seed', default=0, type=int)
# Data
parser.add_argument('--data', default='mnist')
parser.add_argument('--num_examples', default=-1, type=int)
parser.add_argument('--data_seed', default=0, type=int)
parser.add_argument('--sup_sample_weight', default=-1, type=float)
# Optimization
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--epochs', default=500, type=int)
parser.add_argument('--train_bs', default=256, type=int)
parser.add_argument('--test_bs', default=512, type=int)
parser.add_argument('--lr_schedule', default='hat')
parser.add_argument('--lr_warmup', default=10, type=int)
parser.add_argument('--log_each', default=1, type=int)
parser.add_argument('--pretrained', default='')
parser.add_argument('--weight_decay', default=0., type=float)
# Model
parser.add_argument('--model', default='mnist-masked')
parser.add_argument('--conv', default='full')
parser.add_argument('--hh_factors', default=2, type=int)
parser.add_argument('--k', default=4, type=int)
parser.add_argument('--l', default=2, type=int)
parser.add_argument('--hid_dim', type=int, nargs='*', default=[])
# Prior
parser.add_argument('--ssl_model', default='cond-flow')
parser.add_argument('--ssl_dim', default=-1, type=int)
parser.add_argument('--ssl_l', default=2, type=int)
parser.add_argument('--ssl_k', default=3, type=int)
parser.add_argument('--ssl_hd', default=256, type=int)
parser.add_argument('--ssl_conv', default='full')
parser.add_argument('--ssl_hh', default=2, type=int)
parser.add_argument('--ssl_nclasses', default=10, type=int)
# SSL
parser.add_argument('--supervised', default=0, type=int)
parser.add_argument('--sup_weight', default=1., type=float)
parser.add_argument('--cl_weight', default=0, type=float)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# TODO: make it changable
torch.set_num_threads(1)

fmt = {
    'time': '.3f',
}
logger = Logger('logs', base=args.root, fmt=fmt)

# Load data
np.random.seed(args.data_seed)
torch.manual_seed(args.data_seed)
torch.cuda.manual_seed_all(args.data_seed)
trainloader, testloader, data_shape, bits = datautils.load_dataset(args.data, args.train_bs, args.test_bs,
                                                                   seed=args.data_seed, num_examples=args.num_examples,
                                                                   supervised=args.supervised, logs_root=args.root,
                                                                   sup_sample_weight=args.sup_sample_weight)
# Seed for training process
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Create model
dim = int(np.prod(data_shape))
if args.ssl_dim == -1:
    args.ssl_dim = dim
deep_prior = distributions.GaussianDiag(args.ssl_dim)
shallow_prior = distributions.GaussianDiag(dim - args.ssl_dim)
zfprior = distributions.GmmPrior(10, args.ssl_dim)

_, c = np.unique(trainloader.dataset.targets[trainloader.dataset.targets != -1], return_counts=True)
# yprior = torch.distributions.Categorical(probs=torch.FloatTensor(c/c.sum()).to(device))
yprior = flows.ArbitraryConditionalPrior(counts=torch.FloatTensor(c/c.sum()), device=device)
ssl_flow = utils.create_cond_flow(args)
# ssl_flow = torch.nn.DataParallel(ssl_flow.to(device))
ssl_flow.to(device)

# prior = flows.DiscreteConditionalFlowPDF(ssl_flow, deep_prior, yprior, deep_dim=args.ssl_dim,
#                                          shallow_prior=shallow_prior)
# prior = flows.ArbitraryConditionalFlowPDF(ssl_flow, deep_prior, yprior, deep_dim=args.ssl_dim,
#                                          shallow_prior=shallow_prior)
prior = flows.MoGArbitraryConditionalFlowPDF(ssl_flow, deep_prior, yprior, zfprior, deep_dim=args.ssl_dim,
                                         shallow_prior=shallow_prior)

flow = utils.create_flow(args, data_shape)
flow.to(device)
flow = torch.nn.DataParallel(flow.to(device))

model = flows.FlowPDF(flow, prior).to(device)

torch.save(model.state_dict(), os.path.join(args.root, 'model_init.torch'))

parameters = [
    {'params': [p for p in model.parameters() if p.requires_grad], 'weight_decay': args.weight_decay},
]
optimizer = torch.optim.Adamax(parameters, lr=args.lr)
if args.lr_schedule == 'no':
    lr_scheduler = utils.BaseLR(optimizer)
elif args.lr_schedule == 'linear':
    lr_scheduler = utils.LinearLR(optimizer, args.epochs)
elif args.lr_schedule == 'hat':
    lr_scheduler = utils.HatLR(optimizer, args.lr_warmup, args.epochs)
else:
    raise NotImplementedError

if args.pretrained != '':
    model.load_state_dict(torch.load(args.pretrained))
    # model.load_state_dict(torch.load(os.path.join(args.pretrained, 'model.torch')))
    # optimizer.load_state_dict(torch.load(os.path.join(args.pretrained, 'optimizer.torch')))

# Pretrained MNIST classifier
clf = timm.create_model("resnet18", pretrained=False, num_classes=10)
clf.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
clf.load_state_dict(
  torch.hub.load_state_dict_from_url(
    "https://huggingface.co/gpcarl123/resnet18_mnist/resolve/main/resnet18_mnist.pth",
    map_location=device,
    file_name="resnet18_mnist.pth",
  )
)
clf = clf.to(device)

# Dataset
DATA_PATH = './MNIST-SVHN'
test_set = MnistSvhn(data_path = DATA_PATH, split="test", data_multiplication=1, download=True)
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)

print(f"test : {len(test_set)}")


incomplete_dataset = utils.restore_incomplete_dataset(1, '_full')

incomplete_dataloader = DataLoader(
            dataset=incomplete_dataset,
            batch_size=2,
            num_workers=8,
            shuffle=True,
        )

class UniformNoise(object):
    def __init__(self, bits=256):
        self.bits = bits

    def __call__(self, x):
        with torch.no_grad():
            noise = torch.rand_like(x)
            # TODO: generalize. x assumed to be normalized to [0, 1]
            return (x * (self.bits - 1) + noise) / self.bits

    def __repr__(self):
        return "UniformNoise"

apply_noise = UniformNoise()

for i, batch in enumerate(incomplete_dataloader):
    x = apply_noise(batch['data']['mnist'])
    x = x.to(device)
    y = batch['labels']
    y = y.to(device)
    masks = batch['masks']['svhn']
    n_sup = (masks).sum().item()

    log_det, z = model.flow(x)
    y_sup_onehot = torch.nn.functional.one_hot(y.to(x.device), num_classes=len(c))
    y_sup = y_sup_onehot.clone().detach().float().requires_grad_(True)
    model.prior.log_prob(z, y_sup)

    log_prior = torch.ones((x.size(0),)).to(x.device)
    if n_sup != z.shape[0]:
        log_prior[~masks] = model.prior.log_prob(z[~masks])
    if n_sup != 0:
        y_sup_onehot = torch.nn.functional.one_hot(y[masks].to(x.device), num_classes=len(c))
        y_sup = y_sup_onehot.clone().detach().float().requires_grad_(True)
        log_prior[masks] = model.prior.log_prob(z[masks], y=y_sup)
    break
    

correct, count = 0, 0
for i, batch in enumerate(test_dataloader):
    batch = set_inputs_to_device(batch, device='cuda')
    y = batch['labels']
    y_sup_onehot = torch.nn.functional.one_hot(y.to(device), num_classes=len(c))
    y_sup = y_sup_onehot.clone().detach().float().requires_grad_(True)
    x_res = model.conditional_sample(y_sup, device=device)
    correct += sum(clf(x_res).argmax(1) == y_sup.argmax(1)).tolist()
    count += len(y_sup.argmax(1).tolist())

    if i == 0:
        x_test = x_res[:64].detach().cpu().numpy()
        if not np.any(np.isnan(x_test)):
            plt.imshow(utils.viz_array_grid(x_test, 8, 8))
            plt.savefig(f"eval_{model.__class__.__name__}.pdf")

print('Total test accuracy:', correct / count)

# Sample from the GMM prior
x_test = model.zf_sample(device=device).detach().cpu().numpy()
x_test = np.nan_to_num(x_test, nan=0)
plt.imshow(utils.viz_array_grid(x_test, 5, 2))
plt.savefig(f"zf_prior_means_{model.__class__.__name__}.pdf")

def vis_mog(model, ep):
    N = 1000
    samples, labels = model.prior.zfprior.sample([N], labels=True)
    samples = samples.detach().cpu().numpy()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(samples)

    # Step 4: Plot the transformed data
    plt.figure(figsize=(10, 7))
    for label in np.unique(labels):
        plt.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], label=f'Component {label + 1}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Gaussian Mixture')
    plt.savefig(f"PCA_{model.__class__.__name__}_{ep}.pdf")

vis_mog(model, 'eval')