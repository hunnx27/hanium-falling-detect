# 1. python version 버전 체크
python 3.9.0(3.9이하도 이상도 깔면 안됨, 3.9만 깔아야함)
pip 25.2 from c:\users\noname\appdata\local\programs\python\python39\lib\site-packages\pip (python 3.9)

# -- 모델3개(참고 skip)
https://github.com/GajuuzZ/Human-Falling-Detect-Tracks/issues/112

# 2. 라이브러리(requirements.txt) 설치
pip install -r requirements.txt

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

# 3. 그랙피카드 cuda 버전 확인 
``` python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
nvidia-smi
pip list | findstr torch
```
# 4. 추가 설치 pip 업그레이드,  torch,torchvision, cu118 설치(여기서부터 달라지는듯?)
python -m pip install --upgrade pip
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# 5. 설치후 버전 체크 동일한지
``` python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
2.0.1+cu118
11.8
True
```

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

# 6. 실행
python main.py --device cpu


## 노트북 은 다음으로 함 GPU가  다르기 때문
# 4. GeForce 드라이버 설치(nvidia-smi 명령이 되지 않아서 설치함)
장치관리자에서 GeForce드라이버 버전확인(GTX 1050 노트북) 
 https://www.nvidia.com/ko-kr/drivers/
 > GeForce 10 Series(Notebooks) > GeForce GTX 1050 > Windows 11 > Korean
 >> 드라이버 홈 > GeForce GTX 1050 | Windows 11
 >>> 다운로드 및 설치
 >>>> NVIDIA 설정 앱에서 > 드라이버 탭 > GeForce Game Ready 드라이버 추가로 설치

# 5. nvidia-smi 결과(참고 skip)
Thu Oct 16 02:05:54 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 581.57                 Driver Version: 581.57         CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce GTX 1050      WDDM  |   00000000:01:00.0 Off |                  N/A |
| N/A   48C    P0            N/A  / 5001W |       0MiB /   4096MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+

# 6. 잘못설치한 cuda 모두 제거
``` python
pip uninstall torch torchvision torchaudio
```

# 7. 다시 설치
``` python

# 이전 버전꺼 참고 pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# gpt에서 제안한거 pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu121
# torch_stable로 버전명시 버전 pip install torch==2.0.1+cu121 torchvision==0.15.2+cu121 torchaudio==2.0.2+cu121 -f https://download.pytorch.org/whl/torch_stable.html
Looking in links: https://download.pytorch.org/whl/torch_stable.html
ERROR: Could not find a version that satisfies the requirement torch==2.0.1+cu121 (from versions: 1.7.1, 1.7.1+cpu, 1.7.1+cu101, 1.7.1+cu110, 1.8.0, 1.8.0+cpu, 1.8.0+cu101, 1.8.0+cu111, 1.8.1, 1.8.1+cpu, 1.8.1+cu101, 1.8.1+cu102, 1.8.1+cu111, 1.9.0, 1.9.0+cpu, 1.9.0+cu102, 1.9.0+cu111, 1.9.1, 1.9.1+cpu, 1.9.1+cu102, 1.9.1+cu111, 1.10.0, 1.10.0+cpu, 1.10.0+cu102, 1.10.0+cu111, 1.10.0+cu113, 1.10.1, 1.10.1+cpu, 1.10.1+cu102, 1.10.1+cu111, 1.10.1+cu113, 1.10.2, 1.10.2+cpu, 1.10.2+cu102, 1.10.2+cu111, 1.10.2+cu113, 1.11.0, 1.11.0+cpu, 1.11.0+cu113, 1.11.0+cu115, 1.12.0, 1.12.0+cpu, 1.12.0+cu113, 1.12.0+cu116, 1.12.1, 1.12.1+cpu, 1.12.1+cu113, 1.12.1+cu116, 1.13.0, 1.13.0+cpu, 1.13.0+cu116, 1.13.0+cu117, 1.13.1, 1.13.1+cpu, 1.13.1+cu116, 1.13.1+cu117, 2.0.0, 2.0.0+cpu, 2.0.0+cu117, 2.0.0+cu118, 2.0.1, 2.0.1+cpu, 2.0.1+cu117, 2.0.1+cu118, 2.1.0, 2.1.0+cpu, 2.1.0+cu118, 2.1.0+cu121, 2.1.1, 2.1.1+cpu, 2.1.1+cu118, 2.1.1+cu121, 2.1.2, 2.1.2+cpu, 2.1.2+cu118, 2.1.2+cu121, 2.2.0, 2.2.0+cpu, 2.2.0+cu118, 2.2.0+cu121, 2.2.1, 2.2.1+cpu, 2.2.1+cu118, 2.2.1+cu121, 2.2.2, 2.2.2+cpu, 2.2.2+cu118, 2.2.2+cu121, 2.3.0, 2.3.0+cpu, 2.3.0+cu118, 2.3.0+cu121, 2.3.1, 2.3.1+cpu, 2.3.1+cu118, 2.3.1+cu121, 2.4.0, 2.4.1, 2.5.0, 2.5.1, 2.6.0, 2.7.0, 2.7.1, 2.8.0)
ERROR: No matching distribution found for torch==2.0.1+cu121

# 최종 이걸로 함(torch_stable 버전명시버전에 torch버전을 맞춘것)
pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

# 8. 설치 후 torch 버전 확인
``` python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
2.1.0+cu121
12.1
True
```

# 9. 실행
```python
python main.py
```
