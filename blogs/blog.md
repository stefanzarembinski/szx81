## DUKASCOPY Historical Data Feed
https://www.dukascopy.com/swiss/english/marketwatch/historical/

## ???
https://en.wikipedia.org/wiki/Neural_network_(machine_learning)#:~:text=In%20machine%20learning,%20a%20neural%20network
Supervised neural networks that use a mean squared error (MSE) cost function can use formal statistical methods to determine the confidence of the trained model. The MSE on a validation set can be used as an estimate for variance. This value can then be used to calculate the confidence interval of network output, assuming a normal distribution. A confidence analysis made this way is statistically valid as long as the output probability distribution stays the same and the network is not modified.

## Installing PyTorch

### NVIDIA GeForce GT 710 cuda capability problem

  Since version 1.3.1 PyTorch does not support cc < 3.7.

  [NVIDIA GeForce GT 710 cuda capability is 3.5.](https://developer.nvidia.com/install-nsight-visual-studio-edition)

  [How to install pytorch from source for Asus GeForce 710 GT with CUDA CC 3.5 and supported CUDA Toolkit 11.0?](https://forums.developer.nvidia.com/t/how-to-install-pytorch-from-source-for-asus-geforce-710-gt-with-cuda-cc-3-5-and-supported-cuda-toolkit-11-0/147053)

  From Pytorch 1.3.1 on, 3.5 is not supported in the binaries anymore (to reduce the size) 

  See the comment at gpu - Which PyTorch version is CUDA 3.0 compatible? - Stack Overflow 42 and Issues · pytorch/pytorch · GitHub 5. That means I can well use pytorch 1.3.1 and higher with CUDA cc 3.5, I simply have to compile it myself.

  [How to get PyTorch working on Kepler 3.5 GPUs (e.g Tesla K40)](https://medium.com/@jeremistderechte/how-to-get-pytorch-working-on-kepler-3-5-gpus-e-g-tesla-k40-f7275a23b186)


### Installation with CUDA 10.2 and PyTorch 1.2.1

  CUDA 10.2 minimal compability is 3.5.
  PyTorch 1.2.1 minimal compability is 3.5.

  #### Domain Version Compatibility Matrix for PyTorch

  torch   torchvision torchtext torchaudio
  1.10.1	0.11.2	    0.11.1	  0.10.1	  12/15/2021 python36
  1.10	  0.11.0	    0.11.0	  0.10.0	  10/21/2021

  #### CUDA 10.2 installation

  [CUDA Toolkit 10.2 Download](https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal)

  There are CUDA 8.1 and 7 there, as well
  ```
  Installed:
      - Nsight Monitor and HUD Launcher
  Not Installed:
      - Nsight for Visual Studio 2019
        Reason: VS2019 was not foundpython
      - Nsight for Visual Studio 2017
        Reason: VS2017 was not found
      - Nsight for Visual Studio 2015
        Reason: VS2015 was not found
  ```
  #### Efficient and simple installation:

  [conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch](https://medium.com/@khang.pham.exxact/introduction-to-pytorch-with-tutorial-c222b32cd00)
  ```
  torch.__version__
  '1.10.2'
  torch.version.cuda
  '10.2'
  ```
#### Other installation attempts, not realy successful

  [Pytorch from "One-stop solution for easy PyTorch installation":](https://install.pytorch.site/?device=CUDA+10.2&python=Python+3.9)
  ```
  pip3.9 install C:\Temp\torch-1.10.2+cu102-cp39-cp39-win_amd64.whl
  ```
  [torchvision from "Links for torchvision":](https://download.pytorch.org/whl/torchvision/)
  ```
  pip3.9 install C:\Temp\torchvision-0.11.3+cu102-cp39-cp39-win_amd64.whl
  ```
  [torchaudio from Links for torchaudio:](https://download.pytorch.org/whl/torchaudio/)
  ```
  pip3.9 install C:\Temp\torchaudio-0.10.2+cu102-cp39-cp39-win_amd64.whl
  ```
  Installation with python3.9 suffered from wornings, at least. The same with python3.6

  pip install C:\Temp\torch-1.10.2+cu102-cp36-cp36m-win_amd64.whl
  ```
  Processing c:\temp\torch-1.10.2+cu102-cp36-cp36m-win_amd64.whl
  Collecting dataclasses
    Downloading dataclasses-0.8-py3-none-any.whl (19 kB)
  Collecting typing-extensions
    Downloading typing_extensions-4.1.1-py3-none-any.whl (26 kB)
  Installing collected packages: typing-extensions, dataclasses, torch
  Successfully installed dataclasses-0.8 torch-1.10.2+cu102 typing-extensions-4.1.1
  ```

  pip install C:\Temp\torchvision-0.10.1+cu102-cp36-cp36m-win_amd64.whl
  ```
  Processing c:\temp\torchvision-0.10.1+cu102-cp36-cp36m-win_amd64.whl
  Collecting pillow>=5.3.0
    Downloading Pillow-8.4.0-cp36-cp36m-win_amd64.whl (3.2 MB)
      |████████████████████████████████| 3.2 MB 930 kB/s
  Collecting torch==1.9.1
    Downloading torch-1.9.1-cp36-cp36m-win_amd64.whl (222.0 MB)
      |████████████████████████████████| 222.0 MB 1.4 kB/s
  WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='pypi.org', port=443): Read timed out. (read timeout=15)",)': /simple/numpy/
  Collecting numpy
    Downloading numpy-1.19.5-cp36-cp36m-win_amd64.whl (13.2 MB)
      |████████████████████████████████| 13.2 MB 930 kB/s
  Requirement already satisfied: typing-extensions in c:\users\stefa\miniconda3\envs\python36\lib\site-packages (from torch==1.9.1->torchvision==0.10.1+cu102) (4.1.1)
  Requirement already satisfied: dataclasses in c:\users\stefa\miniconda3\envs\python36\lib\site-packages (from torch==1.9.1->torchvision==0.10.1+cu102) (0.8)
  Installing collected packages: torch, pillow, numpy, torchvision
    Attempting uninstall: torch
      Found existing installation: torch 1.10.2+cu102
      Uninstalling torch-1.10.2+cu102:
        Successfully uninstalled torch-1.10.2+cu102
  Successfully installed numpy-1.19.5 pillow-8.4.0 torch-1.9.1 torchvision-0.10.1+cu102 
  ```

## [How to activate conda environment in VS code](https://medium.com/@udiyosovzon/how-to-activate-conda-environment-in-vs-code-ce599497f20d)

## Tutorial
https://www.i32n.com/docs/pytorch/tutorials/beginner/basics/quickstart_tutorial.html

conda install pytorch torchvision torchaudio cudatoolkit=8.0 -c pytorch

# [Building PyTorch from source on Windows to work with an old GPU](https://datagraphi.com/blog/post/2021/9/13/building-pytorch-from-source-on-windows-to-work-with-an-old-gpu)

Update the graphic card driver from [here](https://www.nvidia.com/download/index.aspx).
The updated driver version is 475.14. CUDA 11.x needs >= 452.39.

Install NVIDIA CUDA version 11.8
Installed:
     - Nsight for Visual Studio 2022
     - Nsight Monitor
Not Installed:
     - Nsight for Visual Studio 2019
       Reason: VS2019 was not found
     - Nsight for Visual Studio 2017
       Reason: VS2017 was not found
     - Integrated Graphics Frame Debugger and Profiler
       Reason: see https://developer.nvidia.com/nsight-vstools
     - Integrated CUDA Profilers
       Reason: see https://developer.nvidia.com/nsight-vstools

Graphics Driver version 522.06 This driver could not find compatible hardvare
HD Audio Driver not installed

See curennt CUDA version: nvcc --version

https://s3.amazonaws.com/ossci-windows/magma_2.5.4_cuda118_release.7z
https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z

```
set MAGMA_HOME=D:\Pytorch_requirements\magma
set CMAKE_INCLUDE_PATH=D:\Pytorch_requirements\mkl\include
set LIB=D:\Pytorch_requirements\mkl\lib;%LIB%
set SCCACHE_IDLE_TIMEOUT=0
set TORCH_CUDA_ARCH_LIST=3.5
```
for cuda 11.8 max torch is 2.3.1


git reset --hard HEAD
git clean --force
git pull

git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git rev-parse HEAD
  4a8e49389c33934234dc89616fd17a58e760e2e7

#### Back to the original
git fetch origin
git checkout 4a8e49389c33934234dc89616fd17a58e760e2e7
git reset --hard 4a8e49389c33934234dc89616fd17a58e760e2e7
git clean -d --force
 

git checkout v1.7.0
  HEAD is now at 63d5e9221b [EZ] Pin scipy to 1.12 for Py-3.12 (#127322)
  63d5e9221bedd1546b7d364b5ce4171547db12a9

git fetch origin
git checkout 63d5e9221bedd1546b7d364b5ce4171547db12a9
git reset --hard 63d5e9221bedd1546b7d364b5ce4171547db12a9
git clean -d --force

git submodule sync
git submodule update --init --recursive --jobs 0

