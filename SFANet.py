import torch
import torch.nn as nn
import pandas as pd
import numpy as np

import math
import random

from torch.nn.modules import dropout
import scipy.io as sio
from torch.nn.modules.activation import LeakyReLU
import torchvision
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch.nn.functional as F
import torchvision.transforms as transforms

seed = 1234
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# Load DATA
data = sio.loadmat("F:\备份\HU代码\HyperspectralUnmixing\pooling\SIM1_vca.mat")

abundance_GT = torch.from_numpy(data["A"])  # true abundance
original_HSI = torch.from_numpy(data["Y"])  # mixed abundance
original_Lidar = torch.from_numpy(data["MPN"])  # Lidar
Lidar = torch.from_numpy(data["DSM"])
Lidar_norm = torch.zeros(1, 5, 100, 100)
Lidar = torch.reshape(Lidar, (100, 100))
for i in range(0, 4):
    Lidar_norm[0, i, :, :] = Lidar
# VCA_endmember and GT
VCA_endmember = data["M1"]
GT_endmember = data["M"]

endmember_init = torch.from_numpy(VCA_endmember).unsqueeze(2).unsqueeze(3).float()
GT_init = torch.from_numpy(GT_endmember).unsqueeze(2).unsqueeze(3).float()

band_Number = original_HSI.shape[0]
band_Number_Lidar = original_Lidar.shape[0]
endmember_number, pixel_number = abundance_GT.shape

col = 100

original_HSI = torch.reshape(original_HSI, (band_Number, col, col))
original_Lidar = torch.reshape(original_Lidar, (band_Number_Lidar, col, col))
abundance_GT = torch.reshape(abundance_GT, (endmember_number, col, col))

batch_size = 64
EPOCH = 500

alpha = 0.2
beta = 1
lamda = 0
delta = 0
mu = 0
drop_out = 0.3
learning_rate = 0.004


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


def freeze(layer):
    for child in layer.children():
        for param in child.parameters():
            param.requires_grad = False


# abundance normalization
def norm_abundance_GT(abundance_input, abundance_GT_input):
    abundance_input = abundance_input / (torch.sum(abundance_input, dim=1))
    abundance_input = torch.reshape(
        abundance_input.squeeze(0), (endmember_number, col, col)
    )
    abundance_input = abundance_input.cpu().detach().numpy()
    abundance_GT_input = abundance_GT_input / (torch.sum(abundance_GT_input, dim=0))
    abundance_GT_input = abundance_GT_input.cpu().detach().numpy()
    return abundance_input, abundance_GT_input


# endmember normalization
def norm_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        endmember_input[:, i] = endmember_input[:, i] / np.max(endmember_input[:, i])
        endmember_GT[:, i] = endmember_GT[:, i] / np.max(endmember_GT[:, i])
    return endmember_input, endmember_GT


# plot abundance
def plot_abundance(abundance_input, abundance_GT_input):
    for i in range(0, endmember_number):
        plt.subplot(2, endmember_number, i + 1)
        plt.imshow(abundance_input[i, :, :], cmap="jet")

        plt.subplot(2, endmember_number, endmember_number + i + 1)
        plt.imshow(abundance_GT_input[i, :, :], cmap="jet")
    plt.show()


# plot endmember
def plot_endmember(endmember_input, endmember_GT):
    for i in range(0, endmember_number):
        plt.subplot(1, endmember_number, i + 1)
        plt.plot(endmember_input[:, i], color="b")
        plt.plot(endmember_GT[:, i], color="r")

    plt.show()


# change the index of abundance and endmember
def arange_A_E(abundance_input, abundance_GT_input, endmember_input, endmember_GT):
    RMSE_matrix = np.zeros(endmember_number)
    SAD_matrix = np.zeros(endmember_number)
    SIM1_RMSE_index = [0, 1, 4, 3, 2]
    SIM1_SAD_index = [0, 1, 4, 3, 2]
    # RMSE_index = np.zeros(endmember_number).astype(int)
    # SAD_index = np.zeros(endmember_number).astype(int)
    # RMSE_abundance = np.zeros(endmember_number)
    # SAD_endmember = np.zeros(endmember_number)
    abundance_GT_input[np.arange(endmember_number), :, :] = abundance_GT_input[SIM1_RMSE_index, :, :]
    # abundance_input[np.arange(endmember_number), :, :] = abundance_input[RMSE_index, :, :]
    endmember_GT[:, np.arange(endmember_number)] = endmember_GT[:, SIM1_SAD_index]

    for i in range(0, endmember_number):
        RMSE_matrix[i] = AbundanceRmse(
            abundance_input[i, :, :], abundance_GT_input[i, :, :]
        )
        SAD_matrix[i] = SAD_distance(endmember_input[:, i], endmember_GT[:, i])

        # RMSE_index[i] = np.argmin(RMSE_matrix[i, :])
        # SAD_index[i] = np.argmin(SAD_matrix[i, :])
        # RMSE_abundance[i] = np.min(RMSE_matrix[i, :])
        # SAD_endmember[i] = np.min(SAD_matrix[i, :])
    SIM1_RMSE_index = np.zeros(endmember_number).astype(int)
    SIM1_SAD_index = np.zeros(endmember_number).astype(int)

    # # endmember_input[:, np.arange(endmember_number)] = endmember_input[:, SAD_index]
    RMSE_matrix = RMSE_matrix[:4]
    SAD_matrix = SAD_matrix[:4]
    abundance_input = abundance_input[:4]
    endmember_input = endmember_input[:4]
    abundance_GT_input = abundance_GT_input[:4]
    endmember_GT = endmember_GT[:4]
    return abundance_input, endmember_input, RMSE_matrix, SAD_matrix, abundance_GT_input, endmember_GT


def _aggregate(gate, D, I, K, sort=True):
    if sort:
        _, ind = gate.sort(descending=True)
        gate = gate[:, ind[0, :]]

    U = [(gate[0, i] * D + gate[1, i] * I) for i in range(K)]
    while len(U) != 1:
        temp = []
        for i in range(0, len(U) - 1, 2):
            temp.append(_kronecker_product(U[i], U[i + 1]))
        if len(U) % 2 != 0:
            temp.append(U[-1])
        del U
        U = temp

    return U[0], gate


def _kronecker_product(mat1, mat2):
    return torch.ger(mat1.view(-1), mat2.view(-1)).reshape(*(mat1.size() + mat2.size())).permute(
        [0, 2, 1, 3]).reshape(mat1.size(0) * mat2.size(0), mat1.size(1) * mat2.size(1))


class DGConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True, sort=True,
                 groups=1):
        super(DGConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.register_buffer('D', torch.eye(2))
        self.register_buffer('I', torch.ones(2, 2))

        self.groups = groups
        if groups > 1:
            self.register_buffer('group_mask',
                                 _kronecker_product(torch.ones(out_channels // groups, in_channels // groups),
                                                    torch.eye(groups)))

        if self.out_channels // self.in_channels >= 2:  # Group-up
            self.K = int(np.ceil(math.log2(in_channels)))  # U: [in_channels, in_channels]
            r = int(np.ceil(self.out_channels / self.in_channels))
            _I = _kronecker_product(torch.eye(self.in_channels), torch.ones(r, 1))
            self._I = nn.Parameter(_I, requires_grad=False)
        elif self.in_channels // self.out_channels >= 2:  # Group-down
            self.K = int(np.ceil(math.log2(out_channels)))  # U: [out_channels, out_channels]
            r = int(np.ceil(self.in_channels / self.out_channels))
            _I = _kronecker_product(torch.eye(self.out_channels), torch.ones(1, r))
            self._I = nn.Parameter(_I, requires_grad=False)
        else:
            # in_channels=out_channels, or either one is not the multiple of the other
            self.K = int(np.ceil(math.log2(max(in_channels, out_channels))))

        eps = 1e-8
        gate_init = [eps * random.choice([-1, 1]) for _ in range(self.K)]
        self.register_parameter('gate', nn.Parameter(torch.Tensor(gate_init)))
        self.sort = sort

    def forward(self, x):
        setattr(self.gate, 'org', self.gate.data.clone())
        self.gate.data = ((self.gate.org - 0).sign() + 1) / 2.
        U_regularizer = 2 ** (self.K + torch.sum(self.gate))
        gate = torch.stack((1 - self.gate, self.gate))
        self.gate.data = self.gate.org  # Straight-Through Estimator
        U, gate = _aggregate(gate, self.D, self.I, self.K, sort=self.sort)
        if self.out_channels // self.in_channels >= 2:  # Group-up
            U = torch.mm(self._I, U)
        elif self.in_channels // self.out_channels >= 2:  # Group-down
            U = U[:self.out_channels, :self.out_channels]
            U = torch.mm(U, self._I)

        U = U[:self.out_channels, :self.in_channels]
        if self.groups > 1:
            U = U * self.group_mask
        masked_weight = self.conv.weight * U.view(self.out_channels, self.in_channels, 1, 1)

        x = F.conv2d(x, masked_weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation)
        return x


class load_data(torch.utils.data.Dataset):
    def __init__(self, img, lid, gt, transform=None):
        self.img = img.float()
        self.lid = lid.float()
        self.gt = gt.float()
        self.transform = transform

    def __getitem__(self, idx):
        return self.img, self.lid, self.gt

    def __len__(self):
        return 1


# calculate RMSE of abundance
def AbundanceRmse(inputsrc, inputref):
    img_w = 100
    img_h = 100
    rmse = np.sqrt(((inputsrc - inputref) ** 2).sum() / (img_w * img_h))
    return rmse


# def AbundanceRmse(inputsrc, inputref):
#     rmse = np.sqrt(((inputsrc - inputref) ** 2).mean())
#     return rmse

def conv33(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=1, padding=1)


def conv11(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)


def transconv11(inchannel, outchannel):
    return nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1)


# calculate SAD of endmember
def SAD_distance(src, ref):
    src = src.squeeze()  # 将 src 压缩为一维数组
    ref = ref.squeeze()  # 将 ref 压缩为一维数组
    cos_sim = np.dot(src, ref) / (np.linalg.norm(src) * np.linalg.norm(ref))
    SAD_sim = np.arccos(cos_sim)
    return SAD_sim




class Exponential_Local_Attention_Module(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(Exponential_Local_Attention_Module, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels if inter_channels is not None else in_channels // 2

        if self.inter_channels == 0:
            self.inter_channels = 1

        conv_nd = nn.Conv2d
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.W_z = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
            bn(self.in_channels)
        )

        self.T1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )
        self.T2 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1),
            bn(self.inter_channels),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        batch_size = x1.size(0)

        t1 = self.T1(x1).view(batch_size, self.inter_channels, -1)  # [1, 16, 10000]
        t2 = self.T2(x2).view(batch_size, self.inter_channels, -1)  # [1, 16, 10000]
        t1 = t1.permute(0, 2, 1)  # [1, 10000, 16]

        # Calculate Affinity Matrix
        Affinity_M = torch.matmul(t1, t2)  #

        # Apply exponentiation to enhance feature saliency
        Affinity_M = torch.exp(Affinity_M)


        Affinity_M = Affinity_M.view(100, 100, 10000)


        LAP = F.avg_pool2d(Affinity_M, kernel_size=3, stride=1, padding=1)  # [1, 100, 10000]


        RMP = LAP.max(dim=0, keepdim=True)[0]  #
        CMP = LAP.max(dim=1, keepdim=True)[0].permute(1, 0, 2)   #

        #


        #
        F_concat = torch.cat([RMP, CMP], dim=1)  #

        # Apply Global Average Pooling on the concatenated result to reduce the 200 dimension
        GAP = torch.mean(F_concat, dim=1, keepdim=True)  #

        #
        GAP = GAP.view(1, 100, 100).unsqueeze(0)  #

        # Integrate with x1
        x1 = x1 * GAP.expand_as(x1)  # Element-wise multiplication
        # x1 = torch.cat([x1, GAP], dim=1)  # Concatenate along the channel dimension

        return x1


class conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, dilation, groups, bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.activation(out)

        return out


class SFANet(nn.Module):
    def __init__(self):
        super(SFANet, self).__init__()

        self.activation = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.planes_a = [128, 64, 32]
        self.planes_b = [8, 16, 32]

        # For image a (7×7×input_channels) --> (7×7×planes_a[0])
        self.conv1_a = conv_bn_relu(band_Number, self.planes_a[0], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×input_channels2) --> (7×7×planes_b[0])
        self.conv1_b = conv_bn_relu(band_Number_Lidar, self.planes_b[1], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[0]) --> (7×7×planes_a[1])
        # self.conv2_a = conv_bn_relu(self.planes_a[1], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        self.conv2_a = conv_bn_relu(self.planes_a[0], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[0]) --> (7×7×planes_b[1])
        self.conv2_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=1, bias=True)

        # For image a (7×7×planes_a[1]) --> (7×7×planes_a[2])
        # self.conv3_a = conv_bn_relu(self.planes_a[1], self.planes_a[2], kernel_size=3, padding=1, bias=True)
        # For image b (7×7×planes_b[1]) --> (7×7×planes_b[2])
        # self.conv3_b = conv_bn_relu(self.planes_b[1], self.planes_b[2], kernel_size=3, padding=1, bias=True)

        self.ELAM = Exponential_Local_Attention_Module(in_channels=self.planes_a[2],
                                                       inter_channels=self.planes_a[2] // 2)

        self.FusionLayer = nn.Sequential(

            # nn.Conv2d(self.planes_a[2] * 2, self.planes_a[2] // 2, kernel_size=1, stride=1, padding=0),
            # DGConv2d(self.planes_a[2] * 2, self.planes_a[2] // 2, kernel_size=3, padding=1, groups=1),
            # nn.BatchNorm2d(self.planes_a[2] // 2),
            # nn.ReLU(),

            # nn.Conv2d(self.planes_a[2] // 2, endmember_number, kernel_size=1, stride=1, padding=0),
            DGConv2d(self.planes_a[2], endmember_number, kernel_size=3, padding=1, groups=1),
            nn.BatchNorm2d(endmember_number),
            nn.ReLU(),

            # nn.Conv2d(self.planes_a[2] * 2 // 2, endmember_number, kernel_size=1, stride=1, padding=0)
        )

        self.softmax = nn.Softmax(dim=1)

        # self.fc = nn.Linear(self.planes_a[2], endmember_number)

        self.decoder = nn.Sequential(
            nn.Conv2d(endmember_number, band_Number, kernel_size=1, stride=1, bias=False),
            nn.ReLU(),

        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.conv1_a(x1)
        x2 = self.conv1_b(x2)

        x1 = self.conv2_a(x1)
        x2 = self.conv2_b(x2)

        # x1 = self.conv3_a(x1)
        # x2 = self.conv3_b(x2)

        ss_x1 = self.ELAM(x1, x2)

        x = self.FusionLayer(ss_x1)
        # x = self.avg_pool(x)

        abu = self.softmax(x)
        output = self.decoder(abu)


        return abu, output


# SAD loss of reconstruction
def reconstruction_SADloss(output, target):
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


def SADloss(output, target):
    _, band, h = output.shape
    output = torch.reshape(output, (band, h))
    target = torch.reshape(target, (band, h))
    abundance_loss = torch.acos(torch.cosine_similarity(output, target, dim=0))
    abundance_loss = torch.mean(abundance_loss)

    return abundance_loss


MSE = torch.nn.MSELoss(size_average=True)


# L12+MVC
def l12_norm(inputs):
    out = torch.mean(torch.sum(torch.sqrt(torch.abs(inputs)), dim=1))
    return out


class SparseLoss(nn.Module):
    def __init__(self, sparse_decay):
        super(SparseLoss, self).__init__()
        self.sparse_decay = sparse_decay

    def __call__(self, input):
        loss = l12_norm(input)
        return self.sparse_decay * loss


class MinVolumn(nn.Module):
    def __init__(self, band, endmember_number, delta):
        super(MinVolumn, self).__init__()
        self.band = band
        self.delta = delta
        self.num_classes = endmember_number

    def __call__(self, edm):
        edm_result = torch.reshape(edm, (self.band, self.num_classes))
        edm_mean = edm_result.mean(dim=1, keepdim=True)
        loss = self.delta * ((edm_result - edm_mean) ** 2).sum() / self.band / self.num_classes
        return loss


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-4):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


# EdgeLoss
class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(), k).unsqueeze(0).repeat(5, 1, 1, 1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.cuda()
        self.loss = CharbonnierLoss()
        self.downsampling22 = nn.AvgPool2d(2, 2, ceil_mode=True)

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered = self.conv_gauss(current)  # filter
        down = filtered[:, :, ::2, ::2]  # downsample
        new_filter = torch.zeros_like(filtered)
        new_filter[:, :, ::2, ::2] = down * 4  # upsample
        filtered = self.conv_gauss(new_filter)  # filter
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss


# load data
train_dataset = load_data(
    img=original_HSI, lid=original_Lidar, gt=abundance_GT, transform=transforms.ToTensor()
)
# Data loader
train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=False
)

# net = multiStageUnmixing().cuda()
net = SFANet().cuda()

# # weight init
# net.apply(weights_init)

# decoder weight init by VCA
model_dict = net.state_dict()
model_dict["decoder.0.weight"] = endmember_init

net.load_state_dict(model_dict)

# optimizer 1e-3
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=learning_rate, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)

# train
for epoch in range(EPOCH):
    for i, (x, l, y) in enumerate(train_loader):
        scheduler.step()
        x = x.cuda()
        l = l.cuda()
        net.train().cuda()



        en_abundance, reconstruction_result = net(x, l)

        abundanceLoss = reconstruction_SADloss(x, reconstruction_result)

        MSELoss = MSE(x, reconstruction_result)
        edgeloss = EdgeLoss()
        L21Loss = SparseLoss(lamda)
        # SADloss = SADloss(GT_endmember,net.decoder[0].weight)
        ALoss = abundanceLoss

        BLoss = MSELoss
        # DLoss = SADloss
        # criterionVolumn = MinVolumn(band_Number, endmember_number, delta)

        CLoss = L21Loss(en_abundance)

        # for j in range(0, 5):
        # DLoss = edgeloss(en_abundance, Lidar_norm)
        # DLoss = edgeloss(en_abundance, Lidar_norm.cuda())
        total_loss = (alpha * ALoss) + (beta * BLoss) + CLoss

        optimizer.zero_grad()  # 清空梯度

        total_loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            # 计算端元的SAD和丰度的RMSE
            en_abundance_cpu, abundance_GT_cpu = norm_abundance_GT(en_abundance.cpu(), abundance_GT.cpu())
            decoder_para = net.state_dict()["decoder.0.weight"].cpu().numpy()
            decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember)

            _, _, RMSE_abundance, SAD_endmember, _, _ = arange_A_E(
                en_abundance_cpu, abundance_GT_cpu, decoder_para, GT_endmember
            )

            print(
                "Epoch:", epoch,
                "| loss: %.4f" % total_loss.cpu().data.numpy(),

            )

net.eval()

# en_abundance, reconstruction_result = net(x)

decoder_para = net.state_dict()["decoder.0.weight"].cpu().numpy()
decoder_para = np.mean(np.mean(decoder_para, -1), -1)

en_abundance, abundance_GT = norm_abundance_GT(en_abundance, abundance_GT)
decoder_para, GT_endmember = norm_endmember(decoder_para, GT_endmember)

en_abundance, decoder_para, RMSE_abundance, SAD_endmember, abundance_GT_input, endmember_GT = arange_A_E(
    en_abundance, abundance_GT, decoder_para, GT_endmember
)
print("RMSE",RMSE_abundance.mean())
print("SAD", SAD_endmember.mean())
