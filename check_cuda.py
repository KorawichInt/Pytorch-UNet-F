import torch

print(f"Is avaliable = {torch.cuda.is_available()}")
print(f"Number of cuda device = {torch.cuda.device_count()}")
print(f"Current cuda device no. : {torch.cuda.current_device()}")
print(f"{torch.cuda.device(torch.cuda.current_device())}")
print(f"Device name : {torch.cuda.get_device_name(0)}")