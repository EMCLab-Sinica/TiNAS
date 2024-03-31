import torch

def print_q15(tensor):
    print((tensor*2**15).to(torch.int32))

# For input/output feature maps
def nchw2nhwc(tensor):
    return torch.flatten(torch.permute(tensor, (0, 2, 3, 1)))

# For weights
def nchw2hwnc(tensor):
    return torch.flatten(torch.permute(tensor, (2, 3, 0, 1)))
