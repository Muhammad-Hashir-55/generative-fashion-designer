"""Quick VRAM checker: loads one batch and reports GPU memory usage."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.utils.config import load_config
from src.data.dataloader import create_dataloaders

config = load_config()
loaders = create_dataloaders(config, mode='train')
loader = loaders['train']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', device)
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()

batch = next(iter(loader))
images, labels = batch
print('Batch shapes:', images.shape, labels.shape)
if device.type == 'cuda':
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / (1024**3)
    peak = torch.cuda.max_memory_allocated() / (1024**3)
    print(f'Allocated VRAM: {alloc:.3f} GB')
    print(f'Peak VRAM:      {peak:.3f} GB')
else:
    print('CUDA not available; cannot measure VRAM.')
