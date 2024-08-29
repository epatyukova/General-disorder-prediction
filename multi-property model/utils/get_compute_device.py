import torch


# %%
def get_compute_device(force_cpu=False, prefer_last=True):
    CUDA_available = torch.cuda.is_available()
    MPS_available = torch.backends.mps.is_available() 
    if CUDA_available:
        CUDA_count = torch.cuda.device_count()

    compute_device = torch.device('cpu')

    if force_cpu:
        return compute_device

    elif CUDA_available:
        compute_device = torch.device(f'cuda:{CUDA_count - 1}')
        return compute_device
    
    elif MPS_available:
        compute_device = torch.device("mps")
        return compute_device

    else:
        return compute_device
