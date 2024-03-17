## Getting Started 
### Setting Up the Environment
Driver Version: 546.12  
CUDA Version: 12.3
Windows 11
```
conda create -n ML4IG python=3.8
conda activate ML4IG
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python scripts.1_projection.py
```

## TODO
Batched Tests for self-written codes and batch-less code(a.k.a test for broad casting)  
相机成像的数学原理和数学过程，和代码的结合  
jaxtyping and python typing

## Understanding 3D Scene and Camera
### Camera Obscura
[Video 1](https://www.bilibili.com/video/BV1R4411Q7AR/?spm_id_from=333.337.search-card.all.click&vd_source=75bfab15d56b58245875e3c16f6825ff)  
[Video 2](https://www.youtube.com/watch?v=qvwpDIlN25o)  
[Video 3](https://www.youtube.com/watch?v=-cr5YWZSId0)




