import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def sampling_replacement(n, tau, q):
    samples = np.random.choice(n, size=tau, replace=True, p=q)
    S = np.bincount(samples, minlength=n)
    v = S / (tau * q)
    return v

def sampling_independent(n, p):
    S = np.random.binomial(1, p)
    v = np.zeros(n)
    for i in range(n):
        if S[i] == 1:
            v[i] = 1 / p[i]
    return v

def sampling_tau_nice(n, tau):
    S = np.random.choice(n, size = tau, replace = False)
    v = np.zeros(n)
    for i in S:
        v[i] = n / tau
    return v

# 簡単な2層MLPモデル
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)


n = 8
tau = 10
q = np.array([1/n]*n)
p = np.array([tau / n] * n)

# データローダ（バッチサイズ8）
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=n, shuffle=True)

# モデル・損失・オプティマイザ
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(reduction='none')  # 個別損失を出力
criterion2 = nn.CrossEntropyLoss()

model.train()
history = []

mode = "tau-nice"
for inputs, targets in train_loader:
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()

    outputs = model(inputs)
    loss = criterion(outputs, targets)  # shape: (batch_size,)

    if mode == "replacement":
        v_np = sampling_replacement(n, tau, q)
    elif mode == "independent":
        v_np = sampling_independent(n, p)
    elif mode == "tau-nice":
        v_np = sampling_tau_nice(n, tau)
    else:
        raise ValueError("Unknown sampling mode")

    v = torch.tensor(v_np, dtype = torch.float32, device=device)

    # 重み付き損失合成
    weighted_loss = (loss * v).mean()

    # backward + optimizer
    weighted_loss.backward()
    optimizer.step()

    # # ログ出力
    # print(f"weighted loss: {weighted_loss.item():.4f}")
    # print(f"loss: {loss.mean():.4f}")
    history.append([loss.mean().item(), weighted_loss.item()])

for i, (a, b) in enumerate(history):
    print(f"Step {i+1}: mean loss = {a:.4f}, weighted loss = {b:.4f}")
