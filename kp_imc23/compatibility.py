import torch

def verify_compatiblity():
    print(f'PyTorch version: {torch.__version__}')
    print('*'*10)
    print(f'CUDNN version: {torch.backends.cudnn.version()}')
    print(f'Available GPU devices: {torch.cuda.device_count()}')
    print(f'Device Name: {torch.cuda.get_device_name()}')

    print('CUDA is available' if torch.cuda.is_available() else 'CUDA is not available')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device.type)