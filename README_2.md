# python version
python 3.9.0
pip 25.2 from c:\users\noname\appdata\local\programs\python\python39\lib\site-packages\pip (python 3.9)

# 모델3개
https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/issues/112

# 라이브러리(requirements.txt)
#certifi @ file:///croot/certifi_1671487769961/work/certifi
charset-normalizer==3.3.2
cycler==0.11.0
fonttools==4.38.0
idna==3.7
kiwisolver==1.4.5
matplotlib==3.5.3
numpy==1.21.6
nvidia-cublas-cu11==11.10.3.66
nvidia-cuda-nvrtc-cu11==11.7.99
nvidia-cuda-runtime-cu11==11.7.99
#nvidia-cudnn-cu11==8.5.0.96
nvidia-cudnn-cu11==8.9.4.19
opencv-python==4.10.0.84
packaging==24.0
pandas==1.3.5
Pillow==9.5.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
pytz==2024.1
requests==2.31.0
scipy==1.7.3
six==1.16.0
torch==1.13.1
torchvision==0.14.1
tqdm==4.66.4
typing_extensions==4.7.1
urllib3==1.26.6

# 버전 확인
``` python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
pip list | findstr torch
```
# 추가 설치
python -m pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 설치후 버전
```python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"```
2.0.1+cu118
11.8
True

```nvidia-smi```
Thu Aug 28 23:51:23 2025       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 474.14       Driver Version: 474.14       CUDA Version: 11.4     |
|-------------------------------+----------------------+----------------------+
| GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ... WDDM  | 00000000:01:00.0 N/A |                  N/A |
| 40%    0C    P8    N/A /  N/A |     39MiB /  1024MiB |     N/A      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
|   1  NVIDIA GeForce ... WDDM  | 00000000:07:00.0 N/A |                  N/A |
| 30%    0C    P8    N/A /  N/A |    524MiB /  1024MiB |     N/A      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

# 실행
python main.py --device cpu