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

_, c = np.unique(trainloader.dataset.targets[trainloader.dataset.targets != -1], return_counts=True)
# yprior = torch.distributions.Categorical(probs=torch.FloatTensor(c/c.sum()).to(device))
yprior = flows.ArbitraryConditionalPrior(counts=torch.FloatTensor(c/c.sum()), device=device)
ssl_flow = utils.create_cond_flow(args)
# ssl_flow = torch.nn.DataParallel(ssl_flow.to(device))
ssl_flow.to(device)

# prior = flows.DiscreteConditionalFlowPDF(ssl_flow, deep_prior, yprior, deep_dim=args.ssl_dim,
#                                          shallow_prior=shallow_prior)
prior = flows.ArbitraryConditionalFlowPDF(ssl_flow, deep_prior, yprior, deep_dim=args.ssl_dim,
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

for x, y in trainloader:
    x = x.to(device)
    n_sup = (y != -1).sum().item()

    # hack, because f(...) needed before the g(...)
    model.eval()
    log_det, z = model.flow(x)
    log_prior = torch.ones((x.size(0),)).to(x.device)
    if n_sup != z.shape[0]:
        log_prior[y == -1] = model.prior.log_prob(z[y == -1])
    if n_sup != 0:
        y_sup_onehot = torch.nn.functional.one_hot(y[y != -1].to(x.device), num_classes=len(c))
        y_sup = y_sup_onehot.clone().detach().float().requires_grad_(True)
        log_prior[y != -1] = model.prior.log_prob(z[y != -1], y=y_sup)
    # end of hack, because f(...) needed before the g(...)

    y_test = y_sup[:64]
    numbers = '_'.join(list(map(str, y_test.argmax(1).tolist())))
    x_res = model.conditional_sample(y_test, device=device)
    x_test = x_res.detach().cpu().numpy()
    if not np.any(np.isnan(x_test)):
        plt.imshow(utils.viz_array_grid(x_test, 8, 8))
        plt.savefig(f"train_eval.pdf")
    print('Train accuracy for 64 samples:', sum(clf(x_res).argmax(1) == y_test.argmax(1)).tolist() / 64)
    break

correct, count = 0, 0
for x, y in testloader:
    x = x.to(device)
    y_sup_onehot = torch.nn.functional.one_hot(y[y != -1].to(x.device), num_classes=len(c))
    y_sup = y_sup_onehot.clone().detach().float().requires_grad_(True)
    x_res = model.conditional_sample(y_sup, device=device)
    correct += sum(clf(x_res).argmax(1) == y_sup.argmax(1)).tolist()
    count += len(y_sup.argmax(1).tolist())

print('Total test accuracy:', correct / count)