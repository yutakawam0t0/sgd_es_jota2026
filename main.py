'''Train CIFAR100 with PyTorch.'''
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import argparse
import wandb

from scipy.optimize import nnls
from models import *
from utils import *

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')
parser.add_argument('--batch_size', default=128, type=int, help='training batch size')
parser.add_argument('--optimizer',default="sgd", type=str, help='[momentum,sgd,adam,rmsprop,adagrad,adamw,amsgrad]')
parser.add_argument('--use_wandb', action="store_true", help='Set flag if using wandb.')
parser.add_argument('--sampling', default="normal", type=str, help='[normal,replacement,independent,tau-nice]')
# --- Additional parameters ---
parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
parser.add_argument('--window_size', default=50, type=int, help='Window size for NNLS estimation (W)')
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--scheduler', default='constant', type=str, help='[constant, harmonic, polynomial, cosine]')
parser.add_argument('--poly_alpha', default=0.51, type=float, help='alpha parameter for polynomial decay (0 < alpha < 1)')
args = parser.parse_args()

# --- Fix random seed ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
best_acc = 0
start_epoch = 0
total_epochs = 200

# ==========================
# Data
# ==========================
mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True)

# ---- Evaluation dataloader (without data augmentation) ----
eval_trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_test)
eval_trainloader = torch.utils.data.DataLoader(eval_trainset, batch_size=args.batch_size, shuffle=False, num_workers=2)

# ==========================
# Model & Optimizer
# ==========================
model = ResNet18()
model = model.to(device)

criterion_eval = nn.CrossEntropyLoss()
criterion_train = nn.CrossEntropyLoss(reduction='none')

if args.optimizer == "momentum":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
elif args.optimizer == "adam":
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
elif args.optimizer == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
elif args.optimizer == "rmsprop":
    optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.9)
elif args.optimizer == "adagrad":
    optimizer = optim.Adagrad(model.parameters(), lr=0.01, weight_decay=0)
elif args.optimizer == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999))
elif args.optimizer == "amsgrad":
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)

# ==========================
# Scheduler
# ==========================
if args.scheduler == 'constant':
    scheduler = None
elif args.scheduler == 'harmonic':
    lf_harmonic = lambda epoch: 1.0 / (epoch + 1)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_harmonic)
elif args.scheduler == 'polynomial':
    lf_poly = lambda epoch: 1.0 / ((epoch + 1) ** args.poly_alpha)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf_poly)
elif args.scheduler == 'cosine':
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
else:
    raise ValueError("Unknown scheduler")

tau = 64

# ==========================
# Utility: Gradient norm calculation
# ==========================
def get_minibatch_grad_norm(model):
    grad_list = []
    for p in model.parameters():
        if p.grad is not None:
            grad_list.append(p.grad.detach().view(-1))
    if len(grad_list) == 0:
        return 0.0
    grad_vec = torch.cat(grad_list)
    return torch.norm(grad_vec, p=2).item()

def get_grad_vector(model):
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.detach().view(-1))
    if len(grads) == 0:
        return torch.tensor([0.0], device=device)
    return torch.cat(grads)

snapshots = []

def compare_gradients(epoch):
    g_full, f_full = get_full_grad_and_loss(model, eval_trainloader, device)
    grad_full_norm2 = torch.norm(g_full).item()**2
    
    # Use raw f(x) to prevent NNLS degeneracy
    f_val = f_full
    
    batch_norms = []
    num_samples = 50  # Increased to reduce Monte Carlo noise
    model.train() 
    
    train_iter = iter(trainloader)

    for _ in range(num_samples):
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            train_iter = iter(trainloader)
            inputs, targets = next(train_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        current_bs = inputs.size(0)
        q = np.array([1 / current_bs] * current_bs)
        p = np.array([tau / current_bs] * current_bs)
        
        loss_vec = criterion_train(outputs, targets)
        
        if args.sampling == "normal":
            v_np = np.ones(current_bs)
        elif args.sampling == "replacement":
            v_np = sampling_replacement(current_bs, 32, q)
        elif args.sampling == "independent":
            v_np = sampling_independent(current_bs, p)
        elif args.sampling == "tau-nice":
            v_np = sampling_tau_nice(current_bs, tau)
            
        v = torch.from_numpy(v_np).to(device)
        loss = (loss_vec * v).mean()
        loss.backward()
        
        g_batch = get_grad_vector(model)
        batch_norms.append(torch.norm(g_batch).item()**2)
        
    lhs = np.mean(batch_norms)
    
    snapshots.append((f_val, grad_full_norm2, lhs))
    if len(snapshots) > args.window_size: 
        snapshots.pop(0)
    
    z_hat = {'A': None, 'B': None, 'C': None}
    rhs, residual = None, None
    if len(snapshots) >= 5:
        W = np.stack([
            np.array([2 * s[0] for s in snapshots]),  
            np.array([s[1] for s in snapshots]),      
            np.ones(len(snapshots))                   
        ], axis=1)
        b = np.array([s[2] for s in snapshots])       
        
        z, _ = nnls(W, b)
        z_hat = {'A': float(z[0]), 'B': float(z[1]), 'C': float(z[2])}
        rhs = float(W[-1] @ z)
        residual = float(b[-1] - rhs)
    
    if args.use_wandb:
        current_lr = optimizer.param_groups[0]['lr']
        log_data = {
            'epoch': epoch,
            'lr': current_lr,
            'ES_LHS': lhs,
            'grad_full_norm2': grad_full_norm2,
            'f_val': f_val
        }
        if rhs is not None:
            log_data.update({
                'ES_RHS': rhs,
                'A': z_hat['A'],
                'B': z_hat['B'],
                'C': z_hat['C'],
                'residual': residual
            })
        wandb.log(log_data)

# ==========================
# Training & Test
# ==========================
def train():
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    grad_norms = [] 

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        
        current_bs = inputs.size(0)
        q = np.array([1 / current_bs] * current_bs)
        p = np.array([tau / current_bs] * current_bs)
        
        loss_vec = criterion_train(outputs, targets)

        if args.sampling == "normal":
            v_np = np.ones(current_bs)
        elif args.sampling == "replacement":
            v_np = sampling_replacement(current_bs, 32, q)
        elif args.sampling == "independent":
            v_np = sampling_independent(current_bs, p)
        elif args.sampling == "tau-nice":
            v_np = sampling_tau_nice(current_bs, tau)

        v = torch.from_numpy(v_np).to(device)
        weighted_loss = (loss_vec * v).mean()

        weighted_loss.backward()
        grad_norms.append(get_minibatch_grad_norm(model))
        optimizer.step()

        train_loss += weighted_loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    training_acc = 100. * correct / total
    avg_grad_norm = float(np.mean(grad_norms)) 

    if args.use_wandb:
        wandb.log({
            'epoch': epoch,
            'training_acc': training_acc,
            'training_loss': train_loss / (batch_idx + 1),
            'avg_grad_norm': avg_grad_norm
        })

def test():
    global best_acc
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion_eval(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_loss = total_loss / (batch_idx + 1)
    test_acc = 100. * correct / total
    if args.use_wandb:
        wandb.log({'epoch': epoch, 'test_loss': test_loss, 'test_acc': test_acc})

# ==========================
# Main Loop
# ==========================
if args.use_wandb:
    wandb_project_name = "SGD_ES_3"
    wandb_exp_name = f"{args.optimizer}_{args.scheduler}_{args.sampling}_s{args.seed}_w{args.window_size}"
    wandb.init(config=args, project=wandb_project_name, name=wandb_exp_name)
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

for epoch in range(total_epochs):
    train()
    test()
    compare_gradients(epoch)
    
    if scheduler is not None:
        scheduler.step()