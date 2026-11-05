'''Train Tiny ImageNet with PyTorch.'''
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import zipfile
import requests
import json
from training import train, test
from utils import checkpoint, select_model, get_lr_scheduler, get_bs_scheduler, get_config_value, save_to_csv
from optim.sgd import SGD


class TinyImageNetValDataset(Dataset):
    def __init__(self, annotations_map, img_dir, class_to_idx, transform=None):
        self.annotations_map = annotations_map
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_filenames = list(annotations_map.keys())  # List of image file names

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_path).convert('RGB')

        class_label = self.annotations_map[self.image_filenames[idx]]
        label = self.class_to_idx[class_label]

        if self.transform:
            image = self.transform(image)

        return image, label


# Function to download and extract Tiny ImageNet
def download_and_extract_tiny_imagenet(data_dir='./data'):
    url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
    zip_filename = os.path.join(data_dir, "tiny-imagenet-200.zip")
    extracted_dir = os.path.join(data_dir, "tiny-imagenet-200")

    # Create the directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    # Download the data if it doesn't exist
    if not os.path.exists(extracted_dir):
        print("Downloading Tiny ImageNet dataset...")
        response = requests.get(url, stream=True)
        with open(zip_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        # Extract the data using zipfile
        print("Extracting Tiny ImageNet dataset...")
        with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
            zip_ref.extractall(data_dir)

        print("Tiny ImageNet dataset is ready.")
    else:
        print("Tiny ImageNet dataset already exists.")


def load_validation_annotations():
    annotations_path = './data/tiny-imagenet-200/val/val_annotations.txt'

    # Use a context manager to open the file
    with open(annotations_path, 'r') as file:
        annotations_map = {}

        # Read each line and map the image filename to its class label
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:  # Check for safety to ensure there are enough elements
                image_filename = parts[0]
                class_label = parts[1]
                annotations_map[image_filename] = class_label

    return annotations_map


# Command Line Argument
def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training with Schedulers')
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
    download_and_extract_tiny_imagenet(data_dir='./data')
    mean = (0.4802, 0.4481, 0.3975)
    std = (0.2302, 0.2265, 0.2262)
    transform_train = transforms.Compose([transforms.RandomCrop(64, padding=8),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.RandomRotation(15),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean, std)])
    trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=transform_train)
    class_to_idx = trainset.class_to_idx

    annotations_map = load_validation_annotations()
    val_image_dir = './data/tiny-imagenet-200/val/images'
    transform_test = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean, std)])
    testset = TinyImageNetValDataset(annotations_map=annotations_map, img_dir=val_image_dir, class_to_idx=class_to_idx, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    # Device Setting
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    model_name = get_config_value(config, "model")
    model = select_model(model_name=model_name, num_classes=200).to(device)
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
