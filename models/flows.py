import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from models.normalization import ActNorm, DummyCondActNorm
from models.invertconv import InvertibleConv2d, HHConv2d_1x1, QRInvertibleConv2d, DummyCondInvertibleConv2d
from models.coupling import CouplingLayer, MaskedCouplingLayer, ConditionalCouplingLayer, ConditionalMaskedCouplingLayer
from models.utils import Conv2dZeros, SpaceToDepth, FactorOut, ToLogits, CondToLogits, CondFactorOut, CondSpaceToDepth
from models.utils import DummyCond, IdFunction
import warnings
from pythae.models.base.base_model import BaseEncoder
from models.distributions import GmmPrior

DIMS_RAW = 100
DIMS = 10

class Flow(nn.Module):
    def __init__(self, modules):
        super().__init__()
        self.modules_ = nn.ModuleList(modules)
        self.latent_len = -1
        self.x_shape = -1

    def f(self, x):
        z = None
        log_det_jac = torch.zeros((x.shape[0],)).to(x.device)
        for m in self.modules_:
            x, log_det_jac, z = m(x, log_det_jac, z)
        if z is None:
            z = torch.zeros((x.shape[0], 1))[:, :0].to(x.device)
        self.x_shape = list(x.shape)[1:]
        self.latent_len = z.shape[1]
        z = torch.cat([z, x.reshape((x.shape[0], -1))], dim=1)
        return log_det_jac, z

    def forward(self, x):
        return self.f(x)

    def g(self, z):
        x = z[:, self.latent_len:].view([z.shape[0]] + self.x_shape)
        z = z[:, :self.latent_len]
        for m in reversed(self.modules_):
            x, z = m.g(x, z)
        return x


class ConditionalFlow(Flow):
    def f(self, x, y):
        z = None
        log_det_jac = torch.zeros((x.shape[0],)).to(x.device)
        for m in self.modules_:
            x, log_det_jac, z = m(x, y, log_det_jac, z)
        if z is None:
            z = torch.zeros((x.shape[0], 1))[:, :0].to(x.device)
        self.x_shape = list(x.shape)[1:]
        self.latent_len = z.shape[1]
        z = torch.cat([z, x.reshape((x.shape[0], -1))], dim=1)
        return log_det_jac, z

    def g(self, z, y):
        x = z[:, self.latent_len:].view([z.shape[0]] + self.x_shape)
        z = z[:, :self.latent_len]
        for m in reversed(self.modules_):
            x, z = m.g(x, y, z)
        return x

    def forward(self, x, y):
        return self.f(x, y)


class DiscreteConditionalFlowRaw(ConditionalFlow):
    def __init__(self, modules, num_cat, emb_dim):
        super().__init__(modules)
        self.embeddings = EncoderSVHN(DIMS_RAW)

    def f(self, x, y):
        return super().f(x, self.embeddings(y))

    def g(self, z, y):
        return super().g(z, self.embeddings(y))


class DiscreteConditionalFlowNoEmb(ConditionalFlow):
    def __init__(self, modules, num_cat, emb_dim):
        super().__init__(modules)

    def embeddings(self, y):
        return y

    def f(self, x, y):
        return super().f(x, y)

    def g(self, z, y):
        return super().g(z, y)


class DiscreteConditionalFlow(ConditionalFlow):
    def __init__(self, modules, num_cat, emb_dim):
        super().__init__(modules)
        self.embeddings = nn.Embedding(num_cat, emb_dim)

        self.embeddings.weight.data.zero_()
        l = torch.arange(self.embeddings.weight.data.shape[0])
        self.embeddings.weight.data[l, l] = 1.

    def f(self, x, y):
        return super().f(x, self.embeddings(y))

    def g(self, z, y):
        return super().g(z, self.embeddings(y))


class FlowPDF(nn.Module):
    def __init__(self, flow, prior):
        super().__init__()
        self.flow = flow
        self.prior = prior

    def log_prob(self, x):
        log_det, z = self.flow(x)
        return log_det + self.prior.log_prob(z)
    
    def conditional_sample(self, y, device):
        N = y.shape[0]
        mean1 = self.prior.deep_prior.mean
        sigma1 = torch.exp(self.prior.deep_prior.logsigma)
        z_h = torch.normal(mean1.repeat([N, 1]), sigma1.repeat([N, 1])).to(device)
        z_f = self.prior.flow.g(z_h, y)[:, :, 0, 0]
        mean2 = self.prior.shallow_prior.mean
        sigma2 = torch.exp(self.prior.shallow_prior.logsigma)
        z_aux = torch.normal(mean2.repeat([N, 1]), sigma2.repeat([N, 1])).to(device)
        x = self.flow.module.g(torch.cat([z_aux, z_f], 1))
        return x

    def zf_sample(self, device):
        z_f = self.prior.zfprior.means
        N = z_f.shape[0]
        z_f = self.prior.zfprior.sample([N])
        mean2 = self.prior.shallow_prior.mean
        sigma2 = torch.exp(self.prior.shallow_prior.logsigma)
        z_aux = torch.normal(mean2.repeat([N, 1]), sigma2.repeat([N, 1])).to(device)
        x = self.flow.module.g(torch.cat([z_aux, z_f], 1))
        return x


class DeepConditionalFlowPDF(nn.Module):
    def __init__(self, flow, deep_prior, yprior, deep_dim, shallow_prior=None):
        super().__init__()
        self.flow = flow
        self.shallow_prior = shallow_prior
        self.deep_prior = deep_prior
        self.yprior = yprior
        self.deep_dim = deep_dim

    def log_prob(self, x, y):
        if x.dim() == 2:
            x = x[..., None, None]
        if self.deep_dim == x.shape[1]:
            log_det, z = self.flow(x, y)
            return log_det + self.deep_prior.log_prob(z)
        else:
            log_det, z = self.flow(x[:, -self.deep_dim:], y)
            x_sq = x[:, :-self.deep_dim].squeeze()
            if len(x_sq.shape) == 1:
                x_sq = x_sq.unsqueeze(0)
            return log_det + self.deep_prior.log_prob(z) + self.shallow_prior.log_prob(x_sq)

    def log_prob_joint(self, x, y):
        return self.log_prob(x, y) + self.yprior.log_prob(y)


class ConditionalFlowPDF(nn.Module):
    def __init__(self, flow, prior, emb=True):
        super().__init__()
        self.flow = flow
        self.prior = prior

    def log_prob(self, x, y):
        log_det, z = self.flow(x, y)
        return log_det + self.prior.log_prob(z)


class DiscreteConditionalFlowPDF(DeepConditionalFlowPDF):
    def log_prob_full(self, x):
        sup = self.yprior.enumerate_support().to(x.device)
        logp = []

        n_uniq = sup.size(0)
        y = sup.repeat((x.size(0), 1)).t().reshape((1, -1)).t()[:, 0]
        logp = self.log_prob(x.repeat([n_uniq] + [1]*(len(x.shape)-1)), y)
        return logp.reshape((n_uniq, x.size(0))).t() + self.yprior.log_prob(sup[None])

    def log_prob(self, x, y=None):
        if y is not None:
            return super().log_prob(x, y)
        else:
            logp_joint = self.log_prob_full(x)
            return torch.logsumexp(logp_joint, dim=1)

    def log_prob_posterior(self, x):
        logp_joint = self.log_prob_full(x)
        return logp_joint - torch.logsumexp(logp_joint, dim=1)[:, None]


class ArbitraryConditionalPrior(object):
    def __init__(self, counts, device):
        self.counts = counts
        self.n_classes = counts.shape[0]
        self.conditional = torch.distributions.Categorical(probs=torch.FloatTensor(self.counts/self.counts.sum()).to(device))

    def enumerate_support(self):
        labels_tensor = torch.tensor(range(self.n_classes))
        one_hot_vectors = torch.nn.functional.one_hot(labels_tensor, num_classes=self.n_classes)
        return one_hot_vectors

    def log_prob(self, value):
        return self.conditional.log_prob(value.argmax(axis=1))


class VAEConditionalPrior(object):
    def __init__(self, latent_size, device):
        self.latent_size = latent_size
        self.n_samples = 10
        self.prior = torch.distributions.Normal(torch.zeros((1)).to(device), torch.ones((1)).to(device))

    def enumerate_support(self):
        return torch.normal(0, 1, (self.n_samples, self.latent_size))

    def log_prob(self, value):
        return self.prior.log_prob(value).sum(axis=2)


class RandomConditionalPrior(object):
    def __init__(self, samples, device):
        self.samples = samples
        self.n_samples = samples.shape[0]
        self.device = device
        self.conditional = torch.distributions.Categorical(probs=torch.FloatTensor([1/10]*10).to(self.device))

    def enumerate_support(self):
        return self.samples

    def log_prob(self, value):
        return self.conditional.log_prob(torch.Tensor([1]*value.shape[0]).to(self.device))


class RawConditionalFlowPDF(DiscreteConditionalFlowPDF):
    def __init__(self, flow, deep_prior, yprior, deep_dim, shallow_prior=None):
        super().__init__(flow, deep_prior, yprior, deep_dim, shallow_prior=shallow_prior)
        self.flow = flow
        self.shallow_prior = shallow_prior
        self.deep_prior = deep_prior
        self.yprior = yprior
        self.deep_dim = deep_dim

    def log_prob_full(self, x):
        sup = self.yprior.enumerate_support().to(x.device)

        n_uniq, a, b, c = sup.shape
        N = x.size(0)
        y = sup.unsqueeze(1).repeat((1, N, 1, 1, 1)).reshape((N*n_uniq, a, b, c))
        
        y = y.clone().detach().float().requires_grad_(True)
        logp = self.log_prob(x.repeat([n_uniq] + [1]*(len(x.shape)-1)), y)
        return logp.reshape((n_uniq, x.size(0))).t() + self.yprior.log_prob(sup[None])


class ArbitraryConditionalFlowPDF(DiscreteConditionalFlowPDF):
    def __init__(self, flow, deep_prior, yprior, deep_dim, shallow_prior=None):
        super().__init__(flow, deep_prior, yprior, deep_dim, shallow_prior=shallow_prior)
        self.flow = flow
        self.shallow_prior = shallow_prior
        self.deep_prior = deep_prior
        self.yprior = yprior
        self.deep_dim = deep_dim

    def log_prob_full(self, x):
        sup = self.yprior.enumerate_support().to(x.device)

        n_uniq = sup.size(0)
        N = x.size(0)
        y = sup.repeat((1, N)).reshape((N*n_uniq, -1))
        y = y.clone().detach().float().requires_grad_(True)
        logp = self.log_prob(x.repeat([n_uniq] + [1]*(len(x.shape)-1)), y)
        return logp.reshape((n_uniq, x.size(0))).t() + self.yprior.log_prob(sup[None])


class MoGArbitraryConditionalFlowPDF(DiscreteConditionalFlowPDF):
    def __init__(self, flow, deep_prior, yprior, zfprior, deep_dim, shallow_prior=None):
        super().__init__(flow, deep_prior, yprior, deep_dim, shallow_prior=shallow_prior)
        self.flow = flow
        self.shallow_prior = shallow_prior
        self.deep_prior = deep_prior
        self.yprior = yprior
        self.zfprior = zfprior
        self.deep_dim = deep_dim

    def log_prob_full(self, x):
        x_zf = x[:, -self.deep_dim:].squeeze()
        return self.zfprior.log_prob_full(x_zf)

    def log_prob(self, x, y=None):
        if y is not None:
            return self.log_prob_xy(x, y)
        else:
            return self.log_prob_x(x)

    def log_prob_posterior(self, x):
        logp_joint = self.log_prob_full(x)
        return logp_joint - torch.logsumexp(logp_joint, dim=1)[:, None]

    def log_prob_x(self, x):
        x_pr = x[:, :-self.deep_dim].squeeze()
        x_zf = x[:, -self.deep_dim:].squeeze()
        return self.zfprior.log_prob(x_zf) + self.shallow_prior.log_prob(x_pr)

    def log_prob_xy(self, x, y):
        if x.dim() == 2:
            x = x[..., None, None]
        if self.deep_dim == x.shape[1]:
            log_det, z = self.flow(x, y)
            return log_det + self.deep_prior.log_prob(z) + self.prob_label(x, y.argmax(1))
        else:
            log_det, z = self.flow(x[:, -self.deep_dim:], y)
            x_sq = x[:, :-self.deep_dim].squeeze()
            x_zf = x[:, -self.deep_dim:].squeeze()
            if len(x_sq.shape) == 1:
                x_sq = x_sq.unsqueeze(0)
            if len(x_zf.shape) == 1:
                x_zf = x_zf.unsqueeze(0)
            return log_det + self.deep_prior.log_prob(z) + self.prob_label(x_zf, y.argmax(1)) + self.shallow_prior.log_prob(x_sq)

    def prob_label(self, x, labels):
        pr = self.zfprior.log_prob_full_fast(x)
        return pr[range(x.shape[0]), labels]

    def log_prob_joint(self, x, y):
        return self.log_prob(x, y) + self.yprior.log_prob(y)

    def log_prob_y(self, y, device):
        N = y.shape[0]
        mean1 = self.deep_prior.mean
        sigma1 = torch.exp(self.deep_prior.logsigma)
        z_h = torch.normal(mean1.repeat([N, 1]), sigma1.repeat([N, 1])).to(device)
        z_f = self.flow.g(z_h, y)[:, :, 0, 0]
        return self.prob_label(z_f, y.argmax(1)) + self.yprior.log_prob(y)


class ResNetBlock(nn.Module):
    def __init__(self, channels, use_bn=False):
        super().__init__()
        modules = []
        if use_bn:
            # modules.append(nn.BatchNorm2d(channels))
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3)]
        if use_bn:
            # modules.append(nn.BatchNorm2d(channels))
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3)]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x) + x


class ResNetBlock1x1(nn.Module):
    def __init__(self, channels, use_bn=False):
        super().__init__()
        modules = []
        if use_bn:
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)]
        if use_bn:
            ActNorm(channels, flow=False)
        modules += [
            nn.ReLU(),
            nn.Conv2d(channels, channels, 1)]

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x) + x


def get_resnet1x1(in_channels, channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels * 2
    net = nn.Sequential(
        nn.Conv2d(in_channels, channels, 1, padding=0),
        ResNetBlock1x1(channels, use_bn=True),
        ResNetBlock1x1(channels, use_bn=True),
        ResNetBlock1x1(channels, use_bn=True),
        ResNetBlock1x1(channels, use_bn=True),
        ActNorm(channels, flow=False),
        nn.ReLU(),
        Conv2dZeros(channels, out_channels, 1, padding=0),
    )
    return net


def get_resnet(in_channels, channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels * 2
    net = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, channels, 3),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ActNorm(channels, flow=False),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        Conv2dZeros(channels, out_channels, 3, 0),
    )
    return net


def get_resnet8(in_channels, channels, out_channels=None):
    if out_channels is None:
        out_channels = in_channels * 2
    net = nn.Sequential(
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_channels, channels, 3),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ResNetBlock(channels, use_bn=True),
        ActNorm(channels, flow=False),
        nn.ReLU(),
        nn.ReflectionPad2d(1),
        Conv2dZeros(channels, out_channels, 3, 0),
    )
    return net


def netfunc_for_coupling(in_channels, hidden_channels, out_channels, k=3):
    def foo():
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, k, padding=int(k == 3)),
            nn.ReLU(False),
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.ReLU(False),
            Conv2dZeros(hidden_channels, out_channels, k, padding=int(k == 3))
        )

    return foo


class EncoderSVHN(BaseEncoder):
    def __init__(self, latent_dim):
        super().__init__()
        dataSize = torch.Size([3, 32, 32])
        imgChans = dataSize[0]
        fBase = 32  # base size of filter channels
        self.latent_dim = latent_dim

        self.enc = nn.Sequential(
            # input size: 3 x 32 x 32
            nn.Conv2d(imgChans, fBase, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase) x 16 x 16
            nn.Conv2d(fBase, fBase * 2, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 2) x 8 x 8
            nn.Conv2d(fBase * 2, fBase * 4, 4, 2, 1, bias=True),
            nn.ReLU(True),
            # size: (fBase * 4) x 4 x 4
        )
        self.c1 = nn.Conv2d(fBase * 4, self.latent_dim, 4, 1, 0, bias=True)
        # c1, c2 size: latent_dim x 1 x 1

    def forward(self, x):
        e = self.enc(x)
        emb = self.c1(e)
        a = emb.shape[0]
        emb = emb.squeeze()
        if a == 1:
            return emb.unsqueeze(0)
        return emb


def get_flow(num_layers, k_factor, in_channels=1, hid_dim=[256], conv='full', hh_factors=2,
             cond=False, emb_dim=10, n_cat=10, net='shallow'):
    modules = [
        DummyCond(ToLogits()) if cond else ToLogits(),
    ]
    channels = in_channels

    if conv == 'full':
        convf = lambda x: InvertibleConv2d(x)
    elif conv == 'hh':
        convf = lambda x: HHConv2d_1x1(x, factors=[x]*hh_factors)
    elif conv == 'qr':
        convf = lambda x: QRInvertibleConv2d(x, factors=[x]*hh_factors)
    elif conv == 'qr-abs':
        convf = lambda x: QRInvertibleConv2d(x, factors=[x]*hh_factors, act='no')
    elif conv == 'no':
        convf = lambda x: IdFunction()
    else:
        raise NotImplementedError

    if net == 'shallow':
        couplingnetf = lambda x, y: netfunc_for_coupling(x, hid_dim[0], y)
    elif net == 'resnet':
        couplingnetf = lambda x, y: lambda: get_resnet(x, hid_dim[0], out_channels=y)
    else:
        raise NotImplementedError

    for l in range(num_layers):
        # TODO: FIX
        warnings.warn('==== "get_flow" reduce spatial dimensions only 4 times!!! ====')
        if l != 4:
            if cond:
                modules.append(DummyCond(SpaceToDepth(2)))
            else:
                modules.append(SpaceToDepth(2))
            channels *= 4
        for k in range(k_factor):
            if cond:
                modules.append(DummyCond(ActNorm(channels)))
                modules.append(DummyCond(convf(channels)))
                modules.append(ConditionalCouplingLayer(couplingnetf(channels // 2 + emb_dim, channels)))
            else:
                modules.append(ActNorm(channels))
                modules.append(convf(channels))
                modules.append(CouplingLayer(couplingnetf(channels // 2, channels)))

        if l != num_layers - 1:
            if cond:
                modules.append(DummyCond(FactorOut()))
            else:
                modules.append(FactorOut())

            channels //= 2
            channels -= channels % 2

    return DiscreteConditionalFlow(modules, n_cat, emb_dim) if cond else Flow(modules)


def get_flow_cond(num_layers, k_factor, in_channels=1, hid_dim=256, conv='full', hh_factors=2, num_cat=10, emb_dim=DIMS):
    modules = []
    channels = in_channels
    for l in range(num_layers):
        for k in range(k_factor):
            modules.append(DummyCondActNorm(channels))
            if conv == 'full':
                modules.append(DummyCondInvertibleConv2d(channels))
            elif conv == 'hh':
                modules.append(DummyCond(HHConv2d_1x1(channels, factors=[channels]*hh_factors)))
            elif conv == 'qr':
                modules.append(DummyCond(QRInvertibleConv2d(channels, factors=[channels]*hh_factors)))
            elif conv == 'qr-abs':
                modules.append(DummyCond(QRInvertibleConv2d(channels, factors=[channels]*hh_factors, act='no')))
            else:
                raise NotImplementedError

            netf = lambda: get_resnet1x1(channels//2 + emb_dim, hid_dim, channels)
            modules.append(ConditionalCouplingLayer(netf))

        if l != num_layers - 1:
            modules.append(CondFactorOut())
            channels //= 2
            channels -= channels % 2

    # return DiscreteConditionalFlow(modules, num_cat, emb_dim)
    return DiscreteConditionalFlowNoEmb(modules, num_cat, emb_dim)


def get_flow_raw_cond(num_layers, k_factor, in_channels=1, hid_dim=256, conv='full', hh_factors=2, num_cat=10, emb_dim=DIMS_RAW):
    modules = []
    channels = in_channels
    for l in range(num_layers):
        for k in range(k_factor):
            modules.append(DummyCondActNorm(channels))
            if conv == 'full':
                modules.append(DummyCondInvertibleConv2d(channels))
            elif conv == 'hh':
                modules.append(DummyCond(HHConv2d_1x1(channels, factors=[channels]*hh_factors)))
            elif conv == 'qr':
                modules.append(DummyCond(QRInvertibleConv2d(channels, factors=[channels]*hh_factors)))
            elif conv == 'qr-abs':
                modules.append(DummyCond(QRInvertibleConv2d(channels, factors=[channels]*hh_factors, act='no')))
            else:
                raise NotImplementedError

            netf = lambda: get_resnet1x1(channels//2 + emb_dim, hid_dim, channels)
            modules.append(ConditionalCouplingLayer(netf))

        if l != num_layers - 1:
            modules.append(CondFactorOut())
            channels //= 2
            channels -= channels % 2

    return DiscreteConditionalFlowRaw(modules, num_cat, emb_dim)


def mnist_flow(num_layers=5, k_factor=4, logits=True, conv='full', hh_factors=2, hid_dim=[32, 784]):
    modules = []
    if logits:
        modules.append(ToLogits())

    channels = 1
    hd = hid_dim[0]
    kernel = 3
    for l in range(num_layers):
        if l < 2:
            modules.append(SpaceToDepth(2))
            channels *= 4
        elif l == 2:
            modules.append(SpaceToDepth(7))
            channels *= 49
            hd = hid_dim[1]
            kernel = 1

        for k in range(k_factor):
            modules.append(ActNorm(channels))
            if conv == 'full':
                modules.append(InvertibleConv2d(channels))
            elif conv == 'hh':
                modules.append(HHConv2d_1x1(channels, factors=[channels]*hh_factors))
            elif conv == 'qr':
                modules.append(QRInvertibleConv2d(channels, factors=[channels]*hh_factors))
            elif conv == 'qr-abs':
                modules.append(QRInvertibleConv2d(channels, factors=[channels]*hh_factors, act='no'))
            else:
                raise NotImplementedError
            modules.append(CouplingLayer(netfunc_for_coupling(channels, hd, k=kernel)))

        if l != num_layers - 1:
            modules.append(FactorOut())
            channels //= 2
            channels -= channels % 2

    return Flow(modules)


def mnist_masked_glow(conv='full', hh_factors=2):
    def get_net(in_channels, channels):
        net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, channels, 3),
            ResNetBlock(channels, use_bn=True),
            ResNetBlock(channels, use_bn=True),
            ResNetBlock(channels, use_bn=True),
            ResNetBlock(channels, use_bn=True),
            ActNorm(channels, flow=False),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            Conv2dZeros(channels, in_channels * 2, 3, 0),
        )
        return net

    if conv == 'full':
        convf = lambda x: InvertibleConv2d(x)
    elif conv == 'qr':
        convf = lambda x: QRInvertibleConv2d(x, [x]*hh_factors)
    elif conv == 'hh':
        convf = lambda x: HHConv2d_1x1(x, [x]*hh_factors)
    else:
        raise NotImplementedError

    modules = [
        ToLogits(),
        convf(1),
        MaskedCouplingLayer([1, 28, 28], 'checkerboard0', get_net(1, 64)),
        ActNorm(1),
        convf(1),
        MaskedCouplingLayer([1, 28, 28], 'checkerboard1', get_net(1, 64)),
        ActNorm(1),
        convf(1),
        MaskedCouplingLayer([1, 28, 28], 'checkerboard0', get_net(1, 64)),
        ActNorm(1),
        SpaceToDepth(2),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),

        FactorOut(),

        convf(2),
        MaskedCouplingLayer([2, 14, 14], 'checkerboard0', get_net(2, 64)),
        ActNorm(2),
        convf(2),
        MaskedCouplingLayer([2, 14, 14], 'checkerboard1', get_net(2, 64)),
        ActNorm(2),
        convf(2),
        MaskedCouplingLayer([2, 14, 14], 'checkerboard0', get_net(2, 64)),
        ActNorm(2),
        SpaceToDepth(2),
        convf(8),
        CouplingLayer(lambda: get_net(4, 64)),
        ActNorm(8),
        convf(8),
        CouplingLayer(lambda: get_net(4, 64)),
        ActNorm(8),

        FactorOut(),

        convf(4),
        MaskedCouplingLayer([4, 7, 7], 'checkerboard0', get_net(4, 64)),
        ActNorm(4),
        convf(4),
        MaskedCouplingLayer([4, 7, 7], 'checkerboard1', get_net(4, 64)),
        ActNorm(4),
        convf(4),
        MaskedCouplingLayer([4, 7, 7], 'checkerboard0', get_net(4, 64)),
        ActNorm(4),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),
        convf(4),
        CouplingLayer(lambda: get_net(2, 64)),
        ActNorm(4),
    ]

    return Flow(modules)


def toy2d_flow(conv='full', hh_factors=2, l=5):
    def netf():
        return nn.Sequential(
            nn.Conv2d(1, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 2, 1)
        )

    if conv == 'full':
        convf = lambda x: InvertibleConv2d(x)
    elif conv == 'qr':
        convf = lambda x: QRInvertibleConv2d(x, [x]*hh_factors)
    elif conv == 'hh':
        convf = lambda x: HHConv2d_1x1(x, [x]*hh_factors)
    else:
        raise NotImplementedError

    modules = []
    for _ in range(l):
        modules.append(convf(2))
        modules.append(CouplingLayer(netf))
        modules.append(ActNorm(2))
    return Flow(modules)
