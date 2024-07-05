from math import prod
import sys
import torch
from pythae.models.base.base_model import BaseDecoder, BaseEncoder, ModelOutput
from torch import nn
import timm
import numpy as np
import matplotlib.pyplot as plt
import utils

from multivae.data.datasets import MnistSvhn
from multivae.models.auto_model import AutoConfig, AutoModel
from torch.utils.data import DataLoader
from multivae.data.utils import set_inputs_to_device

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
test_dataloader = DataLoader(test_set, batch_size=256, shuffle=False)


print(f"test : {len(test_set)}")

# Load model from a folder
model_path = sys.argv[1]
model = AutoModel.load_from_folder(model_path).to(device)

correct, count = 0, 0
for i, batch in enumerate(test_dataloader):
    batch = set_inputs_to_device(batch, device='cuda')
    y_sup = batch['labels']
    z = model.encode(inputs=batch, cond='svhn')
    x_mnist = model.decode(embedding=z, modalities='mnist')
    correct += sum(clf(x_mnist['mnist']).argmax(1) == y_sup).tolist()
    count += len(y_sup.tolist())
    if i == 0:
        x_test = x_mnist['mnist'][:64].detach().cpu().numpy()
        if not np.any(np.isnan(x_test)):
            plt.imshow(utils.viz_array_grid(x_test, 8, 8))
            plt.savefig(f"eval_{model.__class__.__name__}.pdf")

print('Total test accuracy:', correct / count)