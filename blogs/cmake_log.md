python setup.py install --cmake
Building wheel torch-2.6.0a0+git4a8e493
-- Building version 2.6.0a0+git4a8e493
cmake -GNinja -DBUILD_PYTHON=True -DBUILD_TEST=True -DCMAKE_BUILD_TYPE=Release -DCMAKE_GENERATOR_TOOLSET_VERSION=14.29 -DCMAKE_INCLUDE_PATH=C:\Users\stefa\Documents\workspaces\pytorch_build\pytorch_requirements\mkl\include;C:\Users\stefa\Documents\workspaces\pytorch_build\pytorch_requirements\sleef\include; -DCMAKE_INSTALL_PREFIX=C:\Users\stefa\Documents\workspaces\pytorch_build\pytorch\torch -DCMAKE_LIBRARY_PATH=C:\Users\stefa\Documents\workspaces\pytorch_build\pytorch_requirements\mkl\lib;C:\Users\stefa\Documents\workspaces\pytorch_build\pytorch_requirements\sleef\lib; -DCMAKE_PREFIX_PATH=C:\Users\stefa\Documents\workspaces\pytorch_build\conda\Lib\site-packages -DJAVA_HOME=C:\Program Files\Java\jdk-19 -DPython_EXECUTABLE=C:\Users\stefa\Documents\workspaces\pytorch_build\conda\python.exe -DTORCH_BUILD_VERSION=2.6.0a0+git4a8e493 -DTORCH_CUDA_ARCH_LIST=3.5 -DUSE_CUDA=1 -DUSE_NUMPY=True -DUSE_ROCM=0 -DUSE_XPU=0 C:\Users\stefa\Documents\workspaces\pytorch_build\pytorch
-- The CXX compiler identification is MSVC 19.29.30156.0
-- The C compiler identification is MSVC 19.29.30156.0
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Detecting C compiler ABI info
-- Detecting C compiler ABI info - done
-- Check for working C compiler: C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe - skipped
-- Detecting C compile features
-- Detecting C compile features - done
-- Not forcing any particular BLAS to be found
CMake Warning at CMakeLists.txt:426 (message):
  TensorPipe cannot be used on Windows.  Set it to OFF


-- Performing Test C_HAS_AVX_1
-- Performing Test C_HAS_AVX_1 - Success
-- Performing Test C_HAS_AVX2_1
-- Performing Test C_HAS_AVX2_1 - Success
-- Performing Test C_HAS_AVX512_1
-- Performing Test C_HAS_AVX512_1 - Success
-- Performing Test CXX_HAS_AVX_1
-- Performing Test CXX_HAS_AVX_1 - Success
-- Performing Test CXX_HAS_AVX2_1
-- Performing Test CXX_HAS_AVX2_1 - Success
-- Performing Test CXX_HAS_AVX512_1
-- Performing Test CXX_HAS_AVX512_1 - Success
-- Current compiler supports avx2 extension. Will build perfkernels.
-- Performing Test CAFFE2_COMPILER_SUPPORTS_AVX512_EXTENSIONS
-- Performing Test CAFFE2_COMPILER_SUPPORTS_AVX512_EXTENSIONS - Success
-- Current compiler supports avx512f extension. Will build fbgemm.
-- Performing Test COMPILER_SUPPORTS_HIDDEN_VISIBILITY
-- Performing Test COMPILER_SUPPORTS_HIDDEN_VISIBILITY - Failed
-- Performing Test COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY
-- Performing Test COMPILER_SUPPORTS_HIDDEN_INLINE_VISIBILITY - Failed
-- Performing Test HAS/UTF_8
-- Performing Test HAS/UTF_8 - Success
-- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8 (found version "11.8")
-- The CUDA compiler identification is NVIDIA 11.8.89
-- Detecting CUDA compiler ABI info
-- Detecting CUDA compiler ABI info - done
-- Check for working CUDA compiler: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe - skipped
-- Detecting CUDA compile features
-- Detecting CUDA compile features - done
-- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include (found version "11.8.89")
-- Caffe2: CUDA detected: 11.8
-- Caffe2: CUDA nvcc is: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe
-- Caffe2: CUDA toolkit directory: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
-- Caffe2: Header version is: 11.8
-- Found Python: C:\Users\stefa\Documents\workspaces\pytorch_build\conda\python.exe (found version "3.12.7") found components: Interpreter
CMake Warning at cmake/public/cuda.cmake:140 (message):
  Failed to compute shorthash for libnvrtc.so
Call Stack (most recent call first):
  cmake/Dependencies.cmake:44 (include)
  CMakeLists.txt:862 (include)


-- Found nvtx3: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/NVTX/c/include
-- Found CUDNN: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cudnn.lib
-- Could NOT find CUSPARSELT (missing: CUSPARSELT_LIBRARY_PATH CUSPARSELT_INCLUDE_PATH)
CMake Warning at cmake/public/cuda.cmake:239 (message):
  Cannot find cuSPARSELt library.  Turning the option off
Call Stack (most recent call first):
  cmake/Dependencies.cmake:44 (include)
  CMakeLists.txt:862 (include)


-- Could NOT find CUDSS (missing: CUDSS_LIBRARY_PATH CUDSS_INCLUDE_PATH)
CMake Warning at cmake/public/cuda.cmake:255 (message):
  Cannot find CUDSS library.  Turning the option off
Call Stack (most recent call first):
  cmake/Dependencies.cmake:44 (include)
  CMakeLists.txt:862 (include)


-- USE_CUFILE is set to 0. Compiling without cuFile support
-- Added CUDA NVCC flags for: -gencode;arch=compute_35,code=sm_35
-- Building using own protobuf under third_party per request.
-- Use custom protobuf build.
CMake Deprecation Warning at third_party/protobuf/cmake/CMakeLists.txt:2 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


--
-- 3.13.0.0
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD
-- Performing Test CMAKE_HAVE_LIBC_PTHREAD - Failed
-- Looking for pthread_create in pthreads
-- Looking for pthread_create in pthreads - not found
-- Looking for pthread_create in pthread
-- Looking for pthread_create in pthread - not found
-- Found Threads: TRUE
-- Caffe2 protobuf include directory: $<BUILD_INTERFACE:C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/protobuf/src>$<INSTALL_INTERFACE:include>
-- Trying to find preferred BLAS backend of choice: MKL
-- MKL_THREADING = OMP
-- Looking for sys/types.h
-- Looking for sys/types.h - found
-- Looking for stdint.h
-- Looking for stdint.h - found
-- Looking for stddef.h
-- Looking for stddef.h - found
-- Check size of void*
-- Check size of void* - done
-- Looking for cblas_sgemm
-- Looking for cblas_sgemm - found
-- Looking for cblas_gemm_bf16bf16f32
-- Looking for cblas_gemm_bf16bf16f32 - found
-- Looking for cblas_gemm_f16f16f32
-- Looking for cblas_gemm_f16f16f32 - not found
-- MKL libraries: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/mkl_intel_lp64_dll.lib;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/mkl_intel_thread_dll.lib;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/mkl_core_dll.lib;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/libiomp5md.lib
-- MKL include directory: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
-- MKL OpenMP type: Intel
-- MKL OpenMP library: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/libiomp5md.lib
CMake Deprecation Warning at third_party/cpuinfo/CMakeLists.txt:1 (CMAKE_MINIMUM_REQUIRED):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at third_party/cpuinfo/deps/clog/CMakeLists.txt:1 (CMAKE_MINIMUM_REQUIRED):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- The ASM compiler identification is MSVC
-- Found assembler: C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe
-- Downloading clog to C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/clog-source (define CLOG_SOURCE_DIR to avoid it)
CMake Warning (dev) at C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/share/cmake-3.27/Modules/ExternalProject.cmake:3136 (message):
  The DOWNLOAD_EXTRACT_TIMESTAMP option was not given and policy CMP0135 is
  not set.  The policy's OLD behavior will be used.  When using a URL
  download, the timestamps of extracted files should preferably be that of
  the time of extraction, otherwise code that depends on the extracted
  contents might not be rebuilt if the URL changes.  The OLD behavior
  preserves the timestamps from the archive instead, but this is usually not
  what you want.  Update your project to the NEW behavior or specify the
  DOWNLOAD_EXTRACT_TIMESTAMP option with a value of true to avoid this
  robustness issue.
Call Stack (most recent call first):
  C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/share/cmake-3.27/Modules/ExternalProject.cmake:4345 (_ep_add_download_command)
  CMakeLists.txt:14 (ExternalProject_Add)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Configuring done (0.5s)
-- Generating done (4.1s)
-- Build files have been written to: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/clog-download
Error: could not load cache
CMake Deprecation Warning at third_party/psimd/CMakeLists.txt:1 (CMAKE_MINIMUM_REQUIRED):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at third_party/FP16/CMakeLists.txt:1 (CMAKE_MINIMUM_REQUIRED):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at third_party/googletest/CMakeLists.txt:4 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at third_party/googletest/googlemock/CMakeLists.txt:45 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at third_party/googletest/googletest/CMakeLists.txt:56 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at third_party/benchmark/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- Found Git: C:/Program Files/Git/cmd/git.exe (found version "2.39.2.windows.1")
-- git Version: v0.0.0
-- Version: 0.0.0
-- Performing Test HAVE_STD_REGEX
-- Performing Test HAVE_STD_REGEX
-- Performing Test HAVE_STD_REGEX -- success
-- Performing Test HAVE_GNU_POSIX_REGEX
-- Performing Test HAVE_GNU_POSIX_REGEX
-- Performing Test HAVE_GNU_POSIX_REGEX -- failed to compile
-- Performing Test HAVE_POSIX_REGEX
-- Performing Test HAVE_POSIX_REGEX
-- Performing Test HAVE_POSIX_REGEX -- failed to compile
-- Performing Test HAVE_STEADY_CLOCK
-- Performing Test HAVE_STEADY_CLOCK
-- Performing Test HAVE_STEADY_CLOCK -- success
CMake Warning (dev) at third_party/fbgemm/CMakeLists.txt:93 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found PythonInterp: C:/Users/stefa/Documents/workspaces/pytorch_build/conda/python.exe (found version "3.12.7")
-- Performing Test COMPILER_SUPPORTS_AVX512
-- Performing Test COMPILER_SUPPORTS_AVX512 - Success
-- Check OMP with lib C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/libiomp5md.lib and flags -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
-- Check OMP with lib C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/libiomp5md.lib and flags -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
CMake Warning (dev) at C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/share/cmake-3.27/Modules/FindPackageHandleStandardArgs.cmake:438 (message):
  The package name passed to `find_package_handle_standard_args` (OpenMP_C)
  does not match the name of the calling package (OpenMP).  This can lead to
  problems in calling code that expects `find_package` result variables
  (e.g., `_FOUND`) to follow a certain pattern.
Call Stack (most recent call first):
  cmake/Modules/FindOpenMP.cmake:590 (find_package_handle_standard_args)
  third_party/fbgemm/CMakeLists.txt:136 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found OpenMP_C: -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
CMake Warning (dev) at C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/share/cmake-3.27/Modules/FindPackageHandleStandardArgs.cmake:438 (message):
  The package name passed to `find_package_handle_standard_args` (OpenMP_CXX)
  does not match the name of the calling package (OpenMP).  This can lead to
  problems in calling code that expects `find_package` result variables
  (e.g., `_FOUND`) to follow a certain pattern.
Call Stack (most recent call first):
  cmake/Modules/FindOpenMP.cmake:590 (find_package_handle_standard_args)
  third_party/fbgemm/CMakeLists.txt:136 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found OpenMP_CXX: -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
-- Found OpenMP: TRUE
CMake Warning at third_party/fbgemm/CMakeLists.txt:138 (message):
  OpenMP found! OpenMP_C_INCLUDE_DIRS =


CMake Warning at third_party/fbgemm/CMakeLists.txt:232 (message):
  ==========


CMake Warning at third_party/fbgemm/CMakeLists.txt:233 (message):
  CMAKE_BUILD_TYPE = Release


CMake Warning at third_party/fbgemm/CMakeLists.txt:234 (message):
  CMAKE_CXX_FLAGS_DEBUG is /Z7 /Ob0 /Od /RTC1 /bigobj


CMake Warning at third_party/fbgemm/CMakeLists.txt:235 (message):
  CMAKE_CXX_FLAGS_RELEASE is /O2 /Ob2 /DNDEBUG /bigobj


CMake Warning at third_party/fbgemm/CMakeLists.txt:236 (message):
  ==========


** AsmJit Summary **
   ASMJIT_DIR=C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/fbgemm/third_party/asmjit
   ASMJIT_TEST=FALSE
   ASMJIT_TARGET_TYPE=SHARED
   ASMJIT_DEPS=
   ASMJIT_LIBS=asmjit
   ASMJIT_CFLAGS=
   ASMJIT_PRIVATE_CFLAGS=-MP;-GF;-Zc:__cplusplus;-Zc:inline;-Zc:strictStrings;-Zc:threadSafeInit-;-W4
   ASMJIT_PRIVATE_CFLAGS_DBG=-GS
   ASMJIT_PRIVATE_CFLAGS_REL=-GS-;-O2;-Oi
CMake Deprecation Warning at third_party/ittapi/CMakeLists.txt:7 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- Using third party subdirectory Eigen.
-- Found Python: C:\Users\stefa\Documents\workspaces\pytorch_build\conda\python.exe (found version "3.12.7") found components: Interpreter Development.Module NumPy
-- Using third_party/pybind11.
-- pybind11 include dirs: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/cmake/../third_party/pybind11/include
-- Could NOT find OpenTelemetryApi (missing: OpenTelemetryApi_INCLUDE_DIRS)
-- Using third_party/opentelemetry-cpp.
-- opentelemetry api include dirs: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/cmake/../third_party/opentelemetry-cpp/api/include
-- Could NOT find MPI_C (missing: MPI_C_LIB_NAMES MPI_C_HEADER_DIR MPI_C_WORKS)
-- Could NOT find MPI_CXX (missing: MPI_CXX_LIB_NAMES MPI_CXX_HEADER_DIR MPI_CXX_WORKS)
-- Could NOT find MPI (missing: MPI_C_FOUND MPI_CXX_FOUND)
CMake Warning at cmake/Dependencies.cmake:956 (message):
  Not compiling with MPI.  Suppress this warning with -DUSE_MPI=OFF
Call Stack (most recent call first):
  CMakeLists.txt:862 (include)


-- Adding OpenMP CXX_FLAGS: -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
-- Will link against OpenMP libraries: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/libiomp5md.lib
-- Found CUB: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include
CMake Deprecation Warning at third_party/gloo/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Warning (dev) at third_party/gloo/CMakeLists.txt:21 (option):
  Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
  --help-policy CMP0077" for policy details.  Use the cmake_policy command to
  set the policy and suppress this warning.

  For compatibility with older versions of CMake, option is clearing the
  normal variable 'BUILD_BENCHMARK'.
This warning is for project developers.  Use -Wno-dev to suppress it.

CMake Warning (dev) at third_party/gloo/CMakeLists.txt:35 (option):
  Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
  --help-policy CMP0077" for policy details.  Use the cmake_policy command to
  set the policy and suppress this warning.

  For compatibility with older versions of CMake, option is clearing the
  normal variable 'USE_NCCL'.
This warning is for project developers.  Use -Wno-dev to suppress it.

CMake Warning (dev) at third_party/gloo/CMakeLists.txt:36 (option):
  Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
  --help-policy CMP0077" for policy details.  Use the cmake_policy command to
  set the policy and suppress this warning.

  For compatibility with older versions of CMake, option is clearing the
  normal variable 'USE_RCCL'.
This warning is for project developers.  Use -Wno-dev to suppress it.

-- MSVC detected
-- Set USE_REDIS OFF
-- Set USE_IBVERBS OFF
-- Set USE_NCCL OFF
-- Set USE_RCCL OFF
-- Set USE_LIBUV ON
-- Only USE_LIBUV is supported on Windows
-- Enabling sccache for CXX
-- Enabling sccache for C
-- Gloo build as SHARED library
CMake Warning (dev) at third_party/gloo/cmake/Cuda.cmake:109 (find_package):
  Policy CMP0074 is not set: find_package uses <PackageName>_ROOT variables.
  Run "cmake --help-policy CMP0074" for policy details.  Use the cmake_policy
  command to set the policy and suppress this warning.

  CMake variable CUDAToolkit_ROOT is set to:

    C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8

  For compatibility, CMake is ignoring the variable.
Call Stack (most recent call first):
  third_party/gloo/cmake/Dependencies.cmake:115 (include)
  third_party/gloo/CMakeLists.txt:111 (include)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found CUDAToolkit: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include (found suitable version "11.8.89", minimum required is "7.0")
-- CUDA detected: 11.8.89
CMake Deprecation Warning at third_party/onnx/CMakeLists.txt:2 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Warning (dev) at third_party/onnx/CMakeLists.txt:107 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

This warning is for project developers.  Use -Wno-dev to suppress it.

Generated: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/third_party/onnx/onnx/onnx_onnx_torch-ml.proto
Generated: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/third_party/onnx/onnx/onnx-operators_onnx_torch-ml.proto
Generated: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/third_party/onnx/onnx/onnx-data_onnx_torch.proto
--
-- ******** Summary ********
--   CMake version                     : 3.27.4
--   CMake command                     : C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/bin/cmake.exe
--   System                            : Windows
--   C++ compiler                      : C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe
--   C++ compiler version              : 19.29.30156.0
--   CXX flags                         : /DWIN32 /D_WINDOWS /GR /EHsc /Zc:__cplusplus /bigobj /FS /utf-8 -DUSE_PTHREADPOOL /EHsc /wd26812
--   Build type                        : Release
--   Compile definitions               : ONNX_ML=1;ONNXIFI_ENABLE_EXT=1;__STDC_FORMAT_MACROS
--   CMAKE_PREFIX_PATH                 : C:\Users\stefa\Documents\workspaces\pytorch_build\conda\Lib\site-packages;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
--   CMAKE_INSTALL_PREFIX              : C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/torch
--   CMAKE_MODULE_PATH                 : C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/cmake/Modules;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/cmake/public/../Modules_CUDA_fix
--
--   ONNX version                      : 1.16.0
--   ONNX NAMESPACE                    : onnx_torch
--   ONNX_USE_LITE_PROTO               : OFF
--   USE_PROTOBUF_SHARED_LIBS          : OFF
--   Protobuf_USE_STATIC_LIBS          : ON
--   ONNX_DISABLE_EXCEPTIONS           : OFF
--   ONNX_DISABLE_STATIC_REGISTRATION  : OFF
--   ONNX_WERROR                       : OFF
--   ONNX_BUILD_TESTS                  : OFF
--   ONNX_BUILD_BENCHMARKS             : OFF
--   ONNX_BUILD_SHARED_LIBS            :
--   BUILD_SHARED_LIBS                 : OFF
--
--   Protobuf compiler                 :
--   Protobuf includes                 :
--   Protobuf libraries                :
--   BUILD_ONNX_PYTHON                 : OFF
-- Found CUDA with FP16 support, compiling with torch.cuda.HalfTensor
-- Adding -DNDEBUG to compile flags
-- Checking prototype magma_get_sgeqrf_nb for MAGMA_V2
-- Checking prototype magma_get_sgeqrf_nb for MAGMA_V2 - False
-- Compiling with MAGMA support
-- MAGMA INCLUDE DIRECTORIES: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/magma/include
-- MAGMA LIBRARIES: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/magma/lib/magma.lib
-- MAGMA V2 check: 0
-- Could not find hardware support for NEON on this machine.
-- No OMAP3 processor on this machine.
-- No OMAP4 processor on this machine.
-- Looking for sbgemm_
-- Looking for sbgemm_ - not found
-- Found a library with LAPACK API (mkl).
disabling ROCM because NOT USE_ROCM is set
-- MIOpen not found. Compiling without MIOpen support
-- MKLDNN_CPU_RUNTIME = OMP
CMake Deprecation Warning at third_party/ideep/mkl-dnn/CMakeLists.txt:17 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- DNNL_TARGET_ARCH: X64
-- DNNL_LIBRARY_NAME: dnnl
CMake Warning (dev) at C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/share/cmake-3.27/Modules/FindPackageHandleStandardArgs.cmake:438 (message):
  The package name passed to `find_package_handle_standard_args` (OpenMP_C)
  does not match the name of the calling package (OpenMP).  This can lead to
  problems in calling code that expects `find_package` result variables
  (e.g., `_FOUND`) to follow a certain pattern.
Call Stack (most recent call first):
  cmake/Modules/FindOpenMP.cmake:590 (find_package_handle_standard_args)
  third_party/ideep/mkl-dnn/cmake/OpenMP.cmake:55 (find_package)
  third_party/ideep/mkl-dnn/CMakeLists.txt:119 (include)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found OpenMP_C: -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
CMake Warning (dev) at C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/share/cmake-3.27/Modules/FindPackageHandleStandardArgs.cmake:438 (message):
  The package name passed to `find_package_handle_standard_args` (OpenMP_CXX)
  does not match the name of the calling package (OpenMP).  This can lead to
  problems in calling code that expects `find_package` result variables
  (e.g., `_FOUND`) to follow a certain pattern.
Call Stack (most recent call first):
  cmake/Modules/FindOpenMP.cmake:590 (find_package_handle_standard_args)
  third_party/ideep/mkl-dnn/cmake/OpenMP.cmake:55 (find_package)
  third_party/ideep/mkl-dnn/CMakeLists.txt:119 (include)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found OpenMP_CXX: -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include
-- Could NOT find Doxyrest (missing: DOXYREST_EXECUTABLE)
CMake Warning (dev) at third_party/ideep/mkl-dnn/cmake/Sphinx.cmake:25 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

Call Stack (most recent call first):
  third_party/ideep/mkl-dnn/cmake/doc.cmake:19 (include)
  third_party/ideep/mkl-dnn/CMakeLists.txt:127 (include)
This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found PythonInterp: C:/Users/stefa/Documents/workspaces/pytorch_build/conda/python.exe (found suitable version "3.12.7", minimum required is "2.7")
-- Could NOT find Sphinx (missing: SPHINX_EXECUTABLE)
-- Enabled workload: TRAINING
-- Enabled primitives: ALL
-- Enabled primitive CPU ISA: ALL
-- Enabled primitive GPU ISA: ALL
-- Enabled GeMM kernels ISA: ALL
-- Primitive cache is enabled
-- The ASM_MASM compiler identification is MSVC
-- Found assembler: C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/ml64.exe
-- Graph component is enabled
-- Graph compiler backend is disabled.
-- Found MKL-DNN: TRUE
-- Version: 10.2.1
-- Build type: Release
-- Using CPU-only version of Kineto
-- Configuring Kineto dependency:
--   KINETO_SOURCE_DIR = C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/kineto/libkineto
--   KINETO_BUILD_TESTS = OFF
--   KINETO_LIBRARY_TYPE = static
CMake Warning (dev) at third_party/kineto/libkineto/CMakeLists.txt:15 (find_package):
  Policy CMP0148 is not set: The FindPythonInterp and FindPythonLibs modules
  are removed.  Run "cmake --help-policy CMP0148" for policy details.  Use
  the cmake_policy command to set the policy and suppress this warning.

This warning is for project developers.  Use -Wno-dev to suppress it.

-- Found PythonInterp: C:/Users/stefa/Documents/workspaces/pytorch_build/conda/python.exe (found version "3.12.7")
INFO CUDA_SOURCE_DIR =
INFO ROCM_SOURCE_DIR =
INFO CUPTI unavailable or disabled - not building GPU profilers
-- Kineto: FMT_SOURCE_DIR = C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/fmt
-- Kineto: FMT_INCLUDE_DIR = C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/fmt/include
INFO CUPTI_INCLUDE_DIR = /extras/CUPTI/include
INFO ROCTRACER_INCLUDE_DIR = /include/roctracer
INFO DYNOLOG_INCLUDE_DIR = C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/kineto/libkineto/third_party/dynolog/
INFO IPCFABRIC_INCLUDE_DIR = C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/third_party/kineto/libkineto/third_party/dynolog//dynolog/src/ipcfabric/
-- Configured Kineto (CPU)
-- Performing Test HAS/WD4624
-- Performing Test HAS/WD4624 - Success
-- Performing Test HAS/WD4068
-- Performing Test HAS/WD4068 - Success
-- Performing Test HAS/WD4067
-- Performing Test HAS/WD4067 - Success
-- Performing Test HAS/WD4267
-- Performing Test HAS/WD4267 - Success
-- Performing Test HAS/WD4661
-- Performing Test HAS/WD4661 - Success
-- Performing Test HAS/WD4717
-- Performing Test HAS/WD4717 - Success
-- Performing Test HAS/WD4244
-- Performing Test HAS/WD4244 - Success
-- Performing Test HAS/WD4804
-- Performing Test HAS/WD4804 - Success
-- Performing Test HAS/WD4273
-- Performing Test HAS/WD4273 - Success
-- Performing Test HAS_WNO_STRINGOP_OVERFLOW
-- Performing Test HAS_WNO_STRINGOP_OVERFLOW - Failed
--
-- Use the C++ compiler to compile (MI_USE_CXX=ON)
--
-- Library base name: mimalloc
-- Version          : 1.8
-- Build type       : release
-- C++ Compiler     : C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe
-- Compiler flags   : /Zc:__cplusplus
-- Compiler defines :
-- Link libraries   : psapi;shell32;user32;advapi32;bcrypt
-- Build targets    : static
--
-- Performing Test HAS_WDEPRECATED
-- Performing Test HAS_WDEPRECATED - Failed
-- don't use NUMA
-- Looking for backtrace
-- Looking for backtrace - not found
-- Could NOT find Backtrace (missing: Backtrace_LIBRARY Backtrace_INCLUDE_DIR)
-- headers outputs:
-- sources outputs:
-- declarations_yaml outputs:
-- Performing Test COMPILER_SUPPORTS_NO_AVX256_SPLIT
-- Performing Test COMPILER_SUPPORTS_NO_AVX256_SPLIT - Failed
-- Using ATen parallel backend: OMP
CMake Deprecation Warning at third_party/sleef/CMakeLists.txt:87 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


CMake Deprecation Warning at third_party/sleef/CMakeLists.txt:91 (cmake_policy):
  The OLD behavior for policy CMP0066 will be removed from a future version
  of CMake.

  The cmake-policies(7) manual explains that the OLD behaviors of all
  policies are deprecated and that a policy should be set to OLD only under
  specific short-term circumstances.  Projects should be ported to the NEW
  behavior and not rely on setting a policy to OLD.


-- Found OpenSSL: C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/lib/libcrypto.lib (found version "3.3.2")
-- Check size of long double
-- Check size of long double - done
-- Performing Test COMPILER_SUPPORTS_FLOAT128
-- Performing Test COMPILER_SUPPORTS_FLOAT128 - Failed
-- Performing Test COMPILER_SUPPORTS_SSE2
-- Performing Test COMPILER_SUPPORTS_SSE2 - Success
-- Performing Test COMPILER_SUPPORTS_SSE4
-- Performing Test COMPILER_SUPPORTS_SSE4 - Success
-- Performing Test COMPILER_SUPPORTS_AVX
-- Performing Test COMPILER_SUPPORTS_AVX - Success
-- Performing Test COMPILER_SUPPORTS_FMA4
-- Performing Test COMPILER_SUPPORTS_FMA4 - Success
-- Performing Test COMPILER_SUPPORTS_AVX2
-- Performing Test COMPILER_SUPPORTS_AVX2 - Success
-- Performing Test COMPILER_SUPPORTS_AVX512F
-- Performing Test COMPILER_SUPPORTS_AVX512F - Success
-- Found OpenMP_C: -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include (found version "2.0")
-- Found OpenMP_CXX: -openmp:experimental -IC:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/include (found version "2.0")
-- Found OpenMP: TRUE (found version "2.0")
-- Performing Test COMPILER_SUPPORTS_OPENMP
-- Performing Test COMPILER_SUPPORTS_OPENMP - Success
-- Performing Test COMPILER_SUPPORTS_WEAK_ALIASES
-- Performing Test COMPILER_SUPPORTS_WEAK_ALIASES - Failed
-- Performing Test COMPILER_SUPPORTS_BUILTIN_MATH
-- Performing Test COMPILER_SUPPORTS_BUILTIN_MATH - Failed
-- Performing Test COMPILER_SUPPORTS_SYS_GETRANDOM
-- Performing Test COMPILER_SUPPORTS_SYS_GETRANDOM - Failed
-- Configuring build for SLEEF-v3.6.0
   Target system: Windows-10.0.22631
   Target processor: AMD64
   Host system: Windows-10.0.22631
   Host processor: AMD64
   Detected C compiler: MSVC @ C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe
   CMake: 3.27.4
   Make program: C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/bin/ninja.exe
-- Using option `/D_CRT_SECURE_NO_WARNINGS  ` to compile libsleef
-- Building shared libs : ON
-- Building static test bins: OFF
-- MPFR : LIB_MPFR-NOTFOUND
-- GMP : LIBGMP-NOTFOUND
-- RT :
-- FFTW3 : LIBFFTW3-NOTFOUND
-- OPENSSL : 3.3.2
-- SDE : SDE_COMMAND-NOTFOUND
-- RUNNING_ON_TRAVIS :
-- COMPILER_SUPPORTS_OPENMP : FALSE
AT_INSTALL_INCLUDE_DIR include/ATen/core
core header install: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/aten/src/ATen/core/TensorBody.h
core header install: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/aten/src/ATen/core/aten_interned_strings.h
core header install: C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/aten/src/ATen/core/enum_tag.h
CMake Deprecation Warning at test/edge/CMakeLists.txt:1 (cmake_minimum_required):
  Compatibility with CMake < 3.5 will be removed from a future version of
  CMake.

  Update the VERSION argument <min> value or use a ...<max> suffix to tell
  CMake that the project does not need compatibility with older versions.


-- Performing Test HAS_WNO_UNUSED_PRIVATE_FIELD
-- Performing Test HAS_WNO_UNUSED_PRIVATE_FIELD - Failed
-- Generating sources for unboxing kernels C:\Users\stefa\Documents\workspaces\pytorch_build\conda\python.exe;-m;torchgen.gen_executorch;--source-path=C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/test/edge/../../test/edge;--install-dir=C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/build/out;--tags-path=C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/test/edge/../../aten/src/ATen/native/tags.yaml;--aten-yaml-path=C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/test/edge/../../aten/src/ATen/native/native_functions.yaml;--use-aten-lib;--op-selection-yaml-path=C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/test/edge/../../test/edge/selected_operators.yaml;--custom-ops-yaml-path=C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/test/edge/../../test/edge/custom_ops.yaml
CMake Warning at CMakeLists.txt:1268 (message):
  Generated cmake files are only fully tested if one builds with system glog,
  gflags, and protobuf.  Other settings may generate files that are not well
  tested.


--
-- ******** Summary ********
-- General:
--   CMake version         : 3.27.4
--   CMake command         : C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/bin/cmake.exe
--   System                : Windows
--   C++ compiler          : C:/Program Files (x86)/Microsoft Visual Studio/2019/BuildTools/VC/Tools/MSVC/14.29.30133/bin/Hostx64/x64/cl.exe
--   C++ compiler id       : MSVC
--   C++ compiler version  : 19.29.30156.0
--   Using ccache if found : OFF
--   CXX flags             : /DWIN32 /D_WINDOWS /GR /EHsc /Zc:__cplusplus /bigobj /FS /utf-8 -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE /wd4624 /wd4068 /wd4067 /wd4267 /wd4661 /wd4717 /wd4244 /wd4804 /wd4273
--   Shared LD flags       : /machine:x64 /ignore:4049 /ignore:4217 /ignore:4099
--   Static LD flags       : /machine:x64 /ignore:4049 /ignore:4217 /ignore:4099
--   Module LD flags       : /machine:x64 /ignore:4049 /ignore:4217 /ignore:4099
--   Build type            : Release
--   Compile definitions   : ONNX_ML=1;ONNXIFI_ENABLE_EXT=1;ONNX_NAMESPACE=onnx_torch;_CRT_SECURE_NO_DEPRECATE=1;IDEEP_USE_MKL;USE_EXTERNAL_MZCRC;MINIZ_DISABLE_ZIP_READER_CRC32_CHECKS;FLASHATTENTION_DISABLE_ALIBI;WIN32_LEAN_AND_MEAN;_UCRT_LEGACY_INFINITY;NOMINMAX;USE_MIMALLOC
--   CMAKE_PREFIX_PATH     : C:\Users\stefa\Documents\workspaces\pytorch_build\conda\Lib\site-packages;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
--   CMAKE_INSTALL_PREFIX  : C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch/torch
--   USE_GOLD_LINKER       : OFF
--
--   TORCH_VERSION         : 2.6.0
--   BUILD_STATIC_RUNTIME_BENCHMARK: OFF
--   BUILD_BINARY          : OFF
--   BUILD_CUSTOM_PROTOBUF : ON
--     Link local protobuf : ON
--   BUILD_PYTHON          : True
--     Python version      : 3.12.7
--     Python executable   : C:\Users\stefa\Documents\workspaces\pytorch_build\conda\python.exe
--     Python library      : C:/Users/stefa/Documents/workspaces/pytorch_build/conda/libs/python312.lib
--     Python includes     : C:/Users/stefa/Documents/workspaces/pytorch_build/conda/include
--     Python site-package : C:\Users\stefa\Documents\workspaces\pytorch_build\conda\Lib\site-packages
--   BUILD_SHARED_LIBS     : ON
--   CAFFE2_USE_MSVC_STATIC_RUNTIME     : OFF
--   BUILD_TEST            : True
--   BUILD_JNI             : OFF
--   BUILD_MOBILE_AUTOGRAD : OFF
--   BUILD_LITE_INTERPRETER: OFF
--   INTERN_BUILD_MOBILE   :
--   TRACING_BASED         : OFF
--   USE_BLAS              : 1
--     BLAS                : mkl
--     BLAS_HAS_SBGEMM     :
--   USE_LAPACK            : 1
--     LAPACK              : mkl
--   USE_ASAN              : OFF
--   USE_TSAN              : OFF
--   USE_CPP_CODE_COVERAGE : OFF
--   USE_CUDA              : 1
--     Split CUDA          :
--     CUDA static link    : OFF
--     USE_CUDNN           : ON
--     USE_CUSPARSELT      : OFF
--     USE_CUDSS           : OFF
--     USE_CUFILE          : OFF
--     CUDA version        : 11.8
--     USE_FLASH_ATTENTION : OFF
--     USE_MEM_EFF_ATTENTION : ON
--     cuDNN version       : 8.9.7
--     CUDA root directory : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8
--     CUDA library        : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cuda.lib
--     cudart library      : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cudart.lib
--     cublas library      : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cublas.lib
--     cufft library       : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cufft.lib
--     curand library      : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/curand.lib
--     cusparse library    : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cusparse.lib
--     cuDNN library       : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cudnn.lib
--     nvrtc               : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/nvrtc.lib
--     CUDA include path   : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/include
--     NVCC executable     : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe
--     CUDA compiler       : C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin/nvcc.exe
--     CUDA flags          :  -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xcompiler  /Zc:__cplusplus -Xcompiler /w -w -Xcompiler /FS -Xfatbin -compress-all -DONNX_NAMESPACE=onnx_torch --use-local-env -gencode arch=compute_35,code=sm_35 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --Werror cross-execution-space-call --no-host-device-move-forward --expt-relaxed-constexpr --expt-extended-lambda  -Xcompiler=/wd4819,/wd4503,/wd4190,/wd4244,/wd4251,/wd4275,/wd4522 -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__
--     CUDA host compiler  :
--     CUDA --device-c     : OFF
--     USE_TENSORRT        :
--   USE_XPU               : 0
--   USE_ROCM              : OFF
--   BUILD_NVFUSER         :
--   USE_EIGEN_FOR_BLAS    :
--   USE_FBGEMM            : ON
--     USE_FAKELOWP          : OFF
--   USE_KINETO            : ON
--   USE_GFLAGS            : OFF
--   USE_GLOG              : OFF
--   USE_LITE_PROTO        : OFF
--   USE_PYTORCH_METAL     : OFF
--   USE_PYTORCH_METAL_EXPORT     : OFF
--   USE_MPS               : OFF
--   USE_MKL               : ON
--   USE_MKLDNN            : ON
--   USE_MKLDNN_ACL        : OFF
--   USE_MKLDNN_CBLAS      : OFF
--   USE_UCC               : OFF
--   USE_ITT               : ON
--   USE_NCCL              : OFF
--   USE_NNPACK            : OFF
--   USE_NUMPY             : ON
--   USE_OBSERVERS         : ON
--   USE_OPENCL            : OFF
--   USE_OPENMP            : ON
--   USE_MIMALLOC          : ON
--   USE_VULKAN            : OFF
--   USE_PROF              : OFF
--   USE_PYTORCH_QNNPACK   : OFF
--   USE_XNNPACK           : ON
--   USE_DISTRIBUTED       : ON
--     USE_MPI               : OFF
--     USE_GLOO              : ON
--     USE_GLOO_WITH_OPENSSL : OFF
--     USE_TENSORPIPE        : OFF
--   Public Dependencies  : caffe2::mkl
--   Private Dependencies : Threads::Threads;pthreadpool;cpuinfo;XNNPACK;fbgemm;ittnotify;fp16;caffe2::openmp;gloo;fmt::fmt-header-only;kineto
--   Public CUDA Deps.    :
--   Private CUDA Deps.   : caffe2::curand;caffe2::cufft;caffe2::cublas;torch::cudnn;gloo_cuda;fmt::fmt-header-only;C:/Users/stefa/Documents/workspaces/pytorch_build/conda/Library/lib/mkl_lapack95_lp64.lib;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/mkl_intel_lp64_dll.lib;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/mkl_intel_thread_dll.lib;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/mkl_core_dll.lib;C:/Users/stefa/Documents/workspaces/pytorch_build/pytorch_requirements/mkl/lib/libiomp5md.lib;C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/lib/x64/cudart_static.lib;CUDA::cusparse;CUDA::cufft;CUDA::cusolver;torch::magma;ATEN_CUDA_FILES_GEN_LIB
--   USE_COREML_DELEGATE     : OFF
--   BUILD_LAZY_TS_BACKEND   : ON
--   USE_ROCM_KERNEL_ASSERT : OFF
-- Performing Test HAS_WMISSING_PROTOTYPES
-- Performing Test HAS_WMISSING_PROTOTYPES - Failed
-- Performing Test HAS_WERROR_MISSING_PROTOTYPES
-- Performing Test HAS_WERROR_MISSING_PROTOTYPES - Failed
-- Configuring done (531.2s)
CMake Error: install(EXPORT "Caffe2Targets" ...) includes target "torch_cpu" which requires target "sleef" that is not in any export set.
-- Generating done (30.4s)
CMake Warning:
  Manually-specified variables were not used by the project:

    CMAKE_GENERATOR_TOOLSET_VERSION
    JAVA_HOME


CMake Generate step failed.  Build files cannot be regenerated correctly.