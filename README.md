# Instant-PINF

```shell
# open `x64 Native Tools Command Prompt`
# activate your virtual environment ( if any )
python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
set TCNN_CUDA_ARCHITECTURES=86
python -m pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
python -m pip install nerfstudio taichi
```
