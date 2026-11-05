import os
import csv


def save_to_csv(csv_path, results_dict):
    headers = {
        "train": ['Epoch', 'Steps', 'Train Loss', 'Train Accuracy', 'Learning Rate'],
        "test": ['Epoch', 'Test Loss', 'Test Accuracy'],
        "norm": ['Epoch', 'Steps', 'Full Gradient Norm'],
        "lr_bs": ['Epoch', 'Steps', 'Learning Rate', 'Batch Size']
    }

    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    for file_type, data in results_dict.items():
        if file_type not in headers:
            raise ValueError(f"Unsupported file_type: {file_type}")

        with open(f"{csv_path}{file_type}.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers[file_type])
            writer.writerows(data)
