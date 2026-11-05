'''Train CIFAR10 with PyTorch.'''
import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import json
from training import train, test
from utils import checkpoint, select_model, get_lr_scheduler, get_bs_scheduler, get_config_value, save_to_csv
from optim.sgd import SGD


# Command Line Argument
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training with Schedulers')
    parser.add_argument('config_path', type=str, help='path of config file(.json)')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--cuda_device', type=int, default=0, help='CUDA device number (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    lr = get_config_value(config, "init_lr")
    epochs = get_config_value(config, "epochs")
    checkpoint_path = config.get("checkpoint_path", "checkpoint.pth.tar")
    csv_path = get_config_value(config, "csv_path")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Dataset Preparation
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Device Setting
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_name = get_config_value(config, "model")
    model = select_model(model_name=model_name, num_classes=10).to(device)
    print(f"model: {model_name}")

    criterion = nn.CrossEntropyLoss()

    optimizer = SGD(model.parameters(), lr=lr)
    bs_scheduler, total_steps = get_bs_scheduler(config, trainset_length=len(trainset))
    print(f"total_steps: {total_steps}")
    lr_scheduler, lr_step_type = get_lr_scheduler(optimizer, config, total_steps)
    print(optimizer)

    # Lists to save results
    train_results = []
    test_results = []
    norm_results = []
    lr_batches = []

    if args.resume:
        checkpoint_ = checkpoint.load(checkpoint_path)
        model.load_state_dict(checkpoint_.get('model_state_dict', model.state_dict()))
        optimizer.load_state_dict(checkpoint_.get('optimizer_state_dict', optimizer.state_dict()))
        lr_scheduler.load_state_dict(checkpoint_.get('lr_scheduler_state_dict', lr_scheduler.state_dict()))
        bs_scheduler.load_state_dict(checkpoint_.get('bs_scheduler_state_dict', bs_scheduler.state_dict()))
        start_epoch = checkpoint_['epoch'] + 1
        train_results = checkpoint_.get('train_results', [])
        test_results = checkpoint_.get('test_results', [])
        norm_results = checkpoint_.get('norm_results', [])
        lr_batches = checkpoint_.get('lr_batches', [])
        steps = checkpoint_['steps']
    else:
        start_epoch = 0
        steps = 0
        batch_size = bs_scheduler.get_batch_size()
        lr_batches.append([1, steps, lr, batch_size])

    for epoch in range(start_epoch, epochs):
        batch_size = bs_scheduler.get_batch_size()
        print(f'batch size: {batch_size}')
        print(f'learning rate: {lr_scheduler.get_last_lr()[0]}')
        steps, lr_batch, norm_result, train_result = train(epoch, steps, model, device, trainset, optimizer, lr_scheduler,
                                                           lr_step_type, criterion, batch_size, cuda=args.cuda_device)
        lr_batches.extend(lr_batch)
        norm_results.append(norm_result)
        train_results.append(train_result)

        test_result = test(epoch, model, device, testloader, criterion)
        test_results.append(test_result)

        if lr_step_type == "epoch":
            lr_scheduler.step()
        bs_scheduler.step()

        checkpoint.save({
            'epoch': epoch,
            'steps': steps,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr_scheduler_state_dict': lr_scheduler.state_dict(),
            'bs_scheduler_state_dict': bs_scheduler.state_dict(),
            'train_results': train_results,
            'test_results': test_results,
            'norm_results': norm_results,
            'lr_batches': lr_batches,
        }, checkpoint_path)

        print(f'Epoch: {epoch + 1}, Steps: {steps}, Train Loss: {train_results[epoch][2]:.4f}, Test Acc: {test_results[epoch][2]:.2f}%')

    # Save to CSV file
    save_to_csv(csv_path, {
        "train": train_results,
        "test": test_results,
        "norm": norm_results,
        "lr_bs": lr_batches
    })
