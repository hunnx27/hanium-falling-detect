import torch
print(torch.__version__)       # 2.0.1+cu118 (또는 설치한 버전)
print(torch.version.cuda)      # 11.8
print(torch.cuda.is_available())  # True