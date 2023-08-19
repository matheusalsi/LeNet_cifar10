import torch

from torch import nn
from torch.nn import Module
import torch.nn.functional as F


import brevitas.nn as qnn
from brevitas.quant import Int8Bias as BiasQuant
from brevitas.quant import Int32Bias

import onnx
from finn.util.test import get_test_model_trained
import brevitas.onnx as bo

class QuantWeightLeNet(Module):
    def __init__(self):
        super(QuantWeightLeNet, self).__init__()
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_bit_width=4)
        self.relu1 = nn.ReLU()
        self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4)
        self.relu2 = nn.ReLU()
        self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4)
        self.relu3 = nn.ReLU()
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = nn.ReLU()
        self.fc3   = qnn.QuantLinear(84, 10, bias=True, weight_bit_width=4)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.size(0), -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

quant_weight_lenet = QuantWeightLeNet()


class QuantWeightActLeNet(Module):
    def __init__(self):
        super(QuantWeightActLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4)
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_bit_width=4)
        self.relu1 = qnn.QuantReLU(bit_width=4)
        self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4)
        self.relu2 = qnn.QuantReLU(bit_width=3)
        self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4)
        self.relu3 = qnn.QuantReLU(bit_width=4)
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4)
        self.relu4 = qnn.QuantReLU(bit_width=4)
        self.fc3   = qnn.QuantLinear(84, 10, bias=True)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

quant_weight_act_lenet = QuantWeightActLeNet()

class QuantWeightActBiasLeNet(Module):
    def __init__(self):
        super(QuantWeightActBiasLeNet, self).__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=4, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(3, 6, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu1 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(6, 16, 5, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu2 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.fc1   = qnn.QuantLinear(16*5*5, 120, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu3 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.fc2   = qnn.QuantLinear(120, 84, bias=True, weight_bit_width=4, bias_quant=Int32Bias)
        self.relu4 = qnn.QuantReLU(bit_width=4, return_quant_tensor=True)
        self.fc3   = qnn.QuantLinear(4, 10, bias=True, weight_bit_width=4, bias_quant=Int32Bias)

    def forward(self, x):
        out = self.quant_inp(x)
        out = self.relu1(self.conv1(out))
        out = F.max_pool2d(out, 2)
        out = self.relu2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.reshape(out.shape[0], -1)
        out = self.relu3(self.fc1(out))
        out = self.relu4(self.fc2(out))
        out = self.fc3(out)
        return out

quant_weight_act_bias_lenet = QuantWeightActBiasLeNet()



# Weight-only model
bo.export_finn_onnx(quant_weight_lenet, torch.randn(1, 3, 32, 32), export_path='finn_4b_weight_lenet.onnx')

# Weight-activation model
bo.export_finn_onnx(quant_weight_act_lenet, torch.randn(1, 3, 32, 32), export_path='finn_4b_weight_act_lenet.onnx')

# Weight-activation-bias model
bo.export_finn_onnx(quant_weight_act_bias_lenet, torch.randn(1, 3, 32, 32), export_path='finn_4b_weight_act_bias_lenet.onnx')