
# [PyTorch Installation From Source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source)

I FAILED!!! THE INSTRUCTIONS ARE USELESS.

## ANACONDA environment
```
conda activate C:\Users\stefa\Documents\workspaces\pytorch_build\conda
cd C:\Users\stefa\Documents\workspaces\pytorch_build
```
## Check CUDA

See `nvcc --version`:
```
  Cuda compilation tools, release 11.8, V11.8.89
  Build cuda_11.8.r11.8/compiler.31833905_0
```

## Get the PyTorch Source

```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
# if you are updating an existing checkout
git submodule sync
git submodule update --init --recursive

cd pytorch
```

## 

## Set environment variables

```
$env:USE_CUDA = "1"
$env:USE_ROCM = "0"
$env:USE_XPU = "0"
$PYTORCH_REQUIREMENTS = "C:\Users\stefa\Documents\workspaces\pytorch_build\pytorch_requirements"
$env:MAGMA_HOME = ($PYTORCH_REQUIREMENTS + "\magma")
$env:CMAKE_INCLUDE_PATH = ($PYTORCH_REQUIREMENTS + "\mkl\include;" + $PYTORCH_REQUIREMENTS + "\sleef\include;")
$env:CMAKE_LIBRARY_PATH = ($PYTORCH_REQUIREMENTS + "\mkl\lib;" + $PYTORCH_REQUIREMENTS + "\sleef\lib;")
$env:LIB = ($PYTORCH_REQUIREMENTS + "\mkl\lib;LIB")
$env:SCCACHE_IDLE_TIMEOUT = "0"
$env:TORCH_CUDA_ARCH_LIST = "3.5"
$env:CMAKE_GENERATOR_TOOLSET_VERSION = "14.29"
# Buildtools: 
# C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64
# $env:DISTUTILS_USE_SDK = "1" offending, rm env:DISTUTILS_USE_SDK
$env:CUDA_VERSION = "11.8.89"

$where = (${env:ProgramFiles(x86)} + "\Microsoft Visual Studio\Installer\vswhere.exe")
foreach ($i in & $where -version [15,17")" -products * -latest -property installationPath) {& ($i + "\VC\Auxiliary\Build\vcvarsall.bat") x64 ("-vcvars_ver=" + $env:CMAKE_GENERATOR_TOOLSET_VERSION)} #Run `vcvarsall.bat`
```

## Install reguirements

```
conda install cmake ninja
# check cmake version: `cmake --version` - has to be >= 3.12
conda install rust
pip install -r requirements.txt
pip install mkl-static mkl-include
```
[Install cuDNN](https://developer.nvidia.com/rdp/cudnn-archive)
Be already registered on Nvidia's page, to download CuDNN. Download a matching version of cuDNN for the CUDA version you earlier installed.
Unzip the downloaded archive file and paste the files from the un-zipped directories to their respective directories of CUDA installed on the system.

Install sccache
[Install scoop on Windows](https://scoop.sh/) first then `scoop install sccache`.

[Install magma](https://s3.amazonaws.com/ossci-windows/magma_2.5.4_cuda101_release.7z)
Store the unzipped result in `$PYTORCH_REQUIREMENTS` folder (magma).

[Install mkl](https://s3.amazonaws.com/ossci-windows/mkl_2020.2.254.7z)
Store the unzipped result in `$PYTORCH_REQUIREMENTS` folder (/mkl).

[Install *sleef*](https://sourceforge.net/projects/sleef/)
Compile it:
```
git clone https://github.com/shibatch/sleef # being in the `$PYTORCH_REQUIREMENTS` folder

cd sleef
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=./ .. # to install in the sleef folder
cmake --build ./ 
```
Resulting `include` and `lib` forders are in the `build` folder.
Place result in `$PYTORCH_REQUIREMENTS + "\sleef"` `include` and `lib` folders if they are not there already.

`vcvarsall.bat` fails to set `CMAKE_CXX_COMPILER` and `CMAKE_C_COMPILER` environmental variables. Perhaps because of backslashes and spaces in windows paths?

si env:CMAKE_CXX_COMPILER '"C:\Program Files (x86)\Microsoft Visual Studio"\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'
si env:CMAKE_C_COMPILER '"C:\Program Files (x86)\Microsoft Visual Studio"\2019\BuildTools\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64\cl.exe'

si env:CMAKE_CXX_COMPILER '"C:/Program Files (x86)/Microsoft Visual Studio"/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe'
si env:"CMAKE_C_COMPILER 'C:/Program Files (x86)/Microsoft Visual Studio"/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe'

python setup.py clean
python setup.py develop

FAILED:
```
-- Performing Test HAS_WMISSING_PROTOTYPES
-- Performing Test HAS_WMISSING_PROTOTYPES - Failed
-- Performing Test HAS_WERROR_MISSING_PROTOTYPES
-- Performing Test HAS_WERROR_MISSING_PROTOTYPES - Failed
-- Configuring incomplete, errors occurred!
```

```