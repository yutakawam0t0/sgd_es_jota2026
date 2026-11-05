import torch
import torch.nn as nn


def get_full_grad_list(model, trainset, optimizer, batch_size, cuda):
    parameters = [p for p in model.parameters()]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() else "cpu")
    full_grad_list = []
    init = True

    for xx, yy in trainloader:
        xx, yy = xx.to(device), yy.to(device)
        optimizer.zero_grad()
        loss = nn.CrossEntropyLoss(reduction='mean')(model(xx), yy)
        loss.backward()

        if init:
            for params in parameters:
                full_grad = torch.zeros_like(params.grad.detach().data)
                full_grad_list.append(full_grad)
            init = False

        for i, params in enumerate(parameters):
            g = params.grad.detach().data
            full_grad_list[i] += (batch_size / len(trainset)) * g

    total_norm = sum(grad.norm().item() ** 2 for grad in full_grad_list) ** 0.5
    return total_norm
