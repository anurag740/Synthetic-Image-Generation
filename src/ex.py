import torch
print(torch.__version__)  # Should print the installed PyTorch version
print("CUDA Available:", torch.cuda.is_available())  # Should be True if CUDA is installed and working
