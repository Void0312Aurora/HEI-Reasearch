import torch
import sys

print(f"Python: {sys.version}")
print(f"Torch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    try:
        t = torch.tensor([1.0]).cuda()
        print("Tensor allocation on GPU: SUCCESS")
    except Exception as e:
        print(f"Tensor allocation on GPU: FAILED - {e}")
else:
    print("CUDA is NOT available.")
