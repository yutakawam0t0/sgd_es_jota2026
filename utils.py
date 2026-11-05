import numpy as np
import torch
import torch.nn as nn

def get_full_grad_and_loss(model, dataloader, device):
    """
    データ拡張なしのデータセットを用いて、正確なフルバッチ勾配と関数値（損失）を計算する。
    """
    was_training = model.training
    model.eval() # 評価中のBatchNorm統計量の更新を防ぐ
    
    full_grad = None
    total_loss = 0.0
    total_samples = 0
    
    # バッチサイズに依存せず正確に足し合わせるため reduction='sum' を使用
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    model.zero_grad()
    for xx, yy in dataloader:
        xx, yy = xx.to(device), yy.to(device)
        outputs = model(xx)
        loss = criterion(outputs, yy)
        loss.backward()
        
        total_loss += loss.item()
        total_samples += len(xx)
        
        # 正確な勾配ベクトルの蓄積
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.detach().view(-1))
        
        batch_grad = torch.cat(grads)
        if full_grad is None:
            full_grad = batch_grad
        else:
            full_grad += batch_grad
            
        model.zero_grad()
        
    model.train(mode=was_training)
    
    # サンプル数で割って平均を出す
    full_grad = full_grad / total_samples
    full_loss = total_loss / total_samples
    
    return full_grad, full_loss

def sampling_replacement(batch_size, roll, q):
    samples = np.random.choice(batch_size, size=roll, replace=True, p=q)
    S = np.bincount(samples, minlength=batch_size)
    v = S / (roll * q)
    return v

def sampling_independent(batch_size, p):
    S = np.random.binomial(1, p)
    v = np.zeros(batch_size)
    for i in range(batch_size):
        if S[i] == 1:
            v[i] = 1 / p[i]
    return v

def sampling_tau_nice(batch_size, tau):
    p = np.array([tau / batch_size] * batch_size)
    return sampling_independent(batch_size, p)