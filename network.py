import torch
import torch.nn as nn
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights

input_size = 8              # PyTouch NetworkモデルのInput次数
output_size = 1             # PyTouch NetworkモデルのOutput次数
weights_init_min = -1       # PyTouch 初期ウエイト下限
weights_init_max = 1        # PyTouch 初期ウエイト上限

device = "cuda" if torch.cuda.is_available() else "cpu"

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, output_size, bias=False).to(device)
        
        # 今回は逆伝搬は必要ない
        self.fc1.weight.requires_grad_(False)
        # ウエイトを,一様分布で初期化
        nn.init.uniform_(self.fc1.weight, a=weights_init_min, b=weights_init_max)

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            x = self.fc1(x)
        return x

class Network2(nn.Module):
    def __init__(self):
        super(Network2, self).__init__()
        self.fc1 = nn.Linear(input_size, 4, bias=True).to(device)
        self.fc2 = nn.Linear(4, output_size, bias=False).to(device)
        self.LRelu = nn.LeakyReLU(0.2).to(device)
        # 今回は逆伝搬は必要ない
        self.fc1.weight.requires_grad_(False)
        self.fc2.weight.requires_grad_(False)
        # ウエイトを,一様分布で初期化
        nn.init.uniform_(self.fc1.weight, a=weights_init_min, b=weights_init_max)

    def forward(self, x):
        with torch.no_grad():
            x = torch.from_numpy(x).float().to(device)
            x = self.fc1(x)
            x = self.LRelu(x)
            x = self.fc2(x)
        return x

if __name__ == "__main__":
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.eval()
    net = model.to(device)
    a = np.zeros((7*3,11*3))
    c = np.array([a, a, a])
    print(c.shape)
    preprocess = weights.transforms()
    img = torch.tensor(c, dtype=torch.int8).to(device)
    # 入力画像への適用
    img_transformed = preprocess(img.unsqueeze(0))
    print(net(img_transformed))