import os
import torch


def save(state, checkpoint_path):
    """Save the model and training state to a checkpoint file."""
    try:
        print(f"Saving checkpoint to {checkpoint_path}...")
        torch.save(state, checkpoint_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")


def load(checkpoint_path):
    """Load the model and training state from a checkpoint file."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path)
        return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
