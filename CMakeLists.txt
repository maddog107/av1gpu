# av1gpu: a CUDA AV1 still-image (AVIF) encoder.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=native]
#   cmake --build build
#
# AV1GPU_CUDA=OFF builds only the CPU parts (bitstream writer, AVIF muxer and the decoder fuzz test),
# which need no CUDA toolkit.
#
# Windows: MSVC (the CUDA host compiler there) with the image libraries from vcpkg; see "Building on Windows" in
# README.md (tools/windows/build.ps1 does all of it).
cmake_minimum_required(VERSION 3.24)

# Default GPU targets (set after enable_language(CUDA), when the CUDA version is known): Turing through
# Blackwell.  Building for one architecture is much faster: -DCMAKE_CUDA_ARCHITECTURES=native (or e.g. 89).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(AV1GPU_DEFAULT_ARCHS ON)
endif()

project(av1gpu VERSION 0.6.0 LANGUAGES C CXX)

option(AV1GPU_CUDA "Build the CUDA encoder" ON)
if(WIN32)
  # the tests use POSIX file APIs and run dav1d / aomdec; on Windows only the encoder is built
  option(AV1GPU_TESTS "Build the tests" OFF)
else()
  option(AV1GPU_TESTS "Build the tests" ON)
endif()
# release binaries: libjpeg, libpng, zlib, libwebp and nvJPEG linked in, so only the NVIDIA driver is needed
# (Windows: the image libraries come static from vcpkg's x64-windows-static triplet, nvJPEG is a DLL)
option(AV1GPU_STATIC_DEPS "Link the image libraries and nvJPEG statically" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  # static C++ runtime (as vcpkg's x64-windows-static and cudart_static), so the executable needs no Visual C++
  # redistributable; UTF-8 sources; M_PI; no min / max macros from windows.h; POSIX names without warnings
  set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
  add_compile_definitions(_USE_MATH_DEFINES NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS _CRT_NONSTDC_NO_WARNINGS)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:/utf-8>" "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
  set(CMAKE_CXX_FLAGS_RELEASE "/O2")
  # 8 MB thread stacks as on Linux (std::thread takes the executable's default, 1 MB unless set)
  add_link_options(/STACK:8388608)
  set(AV1GPU_CXX_WARNINGS "")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
  set(AV1GPU_CXX_WARNINGS -Wall -Wno-missing-braces -Wno-unused-function)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------------------------------
# av1core: decision record -> AV1 bitstream (CPU writer), AVIF muxer, shared DSP headers
add_library(av1core STATIC av1core/src/encoder.cpp av1core/src/avif.cpp)
target_include_directories(av1core PUBLIC av1core/include av1core/src)
if(NOT MSVC)
  target_compile_options(av1core PRIVATE -Wall -Wno-missing-braces -Wno-unused-variable -Wno-unused-function)
endif()

if(AV1GPU_TESTS)
  enable_testing()
  # random legal decision records -> av1core -> dav1d + aomdec must reproduce the reconstruction exactly
  add_executable(av1g_fuzz tests/fuzz.cpp)
  target_link_libraries(av1g_fuzz av1core)
  add_test(NAME fuzz_dav1d_aomdec COMMAND av1g_fuzz --seed 1 --count 200 --quiet)
  # lossless only: closed-loop records whose reconstruction must equal their random source picture
  add_test(NAME fuzz_lossless COMMAND av1g_fuzz --seed 1000000 --count 120 --lossless 100 --quiet)
  # AVIF with an alpha auxiliary image: box layout (libavif 1.4's), item payloads decoded by dav1d + aomdec
  add_executable(avif_alpha_test tests/avif_alpha_test.cpp)
  target_link_libraries(avif_alpha_test av1core)
  add_test(NAME avif_alpha COMMAND avif_alpha_test)
endif()

if(NOT AV1GPU_CUDA)
  return()
endif()

# ---------------------------------------------------------------------------------------------------
enable_language(CUDA)
# the toolkit CMake found for the CUDA compiler (with Clang, CMAKE_CUDA_COMPILER_VERSION is Clang's version)
find_package(CUDAToolkit 12.0 REQUIRED)
if(AV1GPU_DEFAULT_ARCHS)
  if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;90-virtual")
  endif()
  if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    # Clang has no PTX-only targets (CMake warns): the newest architecture gets SASS + PTX instead ("120")
    string(REGEX REPLACE "-real;[0-9]+-virtual$" "" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  endif()
endif()
message(STATUS "CUDA ${CUDAToolkit_VERSION} (${CMAKE_CUDA_COMPILER_ID} ${CMAKE_CUDA_COMPILER_VERSION}), architectures: ${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
if(CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA")
  string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -lineinfo")
endif()
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
if(CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
  # Clang (-DCMAKE_CUDA_COMPILER=clang++; pick the toolkit with -DCUDAToolkit_ROOT): it still compiles with
  # a toolkit newer than it knows, but would warn about that in every file
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Wno-unknown-cuda-version>)
endif()
# nvcc only: constexpr host functions (std::min, ...) callable from device code; Clang allows that by default
set(AV1GPU_CUDA_OPTIONS $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--expt-relaxed-constexpr>)
if(AV1GPU_STATIC_DEPS AND NOT WIN32)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  set(AV1GPU_NVJPEG CUDA::nvjpeg_static)
else()
  # (the Windows toolkit has no static nvJPEG: nvjpeg64_*.dll goes next to av1gpu.exe)
  set(AV1GPU_NVJPEG CUDA::nvjpeg)
endif()
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
if(WIN32)
  find_package(WebP CONFIG REQUIRED)   # vcpkg's libwebp
  set(AV1GPU_WEBP WebP::webp)
else()
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)
  set(AV1GPU_WEBP PkgConfig::WEBP)
endif()
find_package(Threads REQUIRED)

# av1core GPU backend: symbolization + entropy coding, deblocking + CDEF
add_library(av1core_gpu STATIC av1core/src/gpu_entropy.cu av1core/src/gpu_postfilter.cu)
target_link_libraries(av1core_gpu PUBLIC av1core CUDA::cudart_static)
# per-thread default stream: the entropy thread's batches run concurrently with the next encode.  Clang's driver
# passes -fgpu-default-stream on for HIP only (for CUDA it warns "argument unused"), so it goes to cc1 directly,
# which then does what nvcc does (cudaLaunchKernel_ptsz for <<<>>>, the _ptds runtime API)
target_compile_options(av1core_gpu PRIVATE ${AV1GPU_CUDA_OPTIONS}
  $<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:--default-stream per-thread>
  "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:Clang>>:SHELL:-Xclang -fgpu-default-stream=per-thread>")

# av1f: libaom tune=iq decisions, variance boost, loop-filter / CDEF strength search
add_library(av1f STATIC av1f/src/varboost.cu av1f/src/lfsearch.cu)
target_include_directories(av1f PUBLIC av1f/include)
target_link_libraries(av1f PUBLIC av1core CUDA::cudart_static)
target_compile_options(av1f PRIVATE ${AV1GPU_CUDA_OPTIONS})

# ingest: JPEG -> GPU-resident 4:2:0 planes (nvJPEG hardware/hybrid decode or CPU entropy decode + GPU IDCT)
add_library(gpujpegingest STATIC ingest/src/jpeg_reader.cpp ingest/src/kernels.cu ingest/src/pipeline.cu)
target_include_directories(gpujpegingest PUBLIC ingest/src)
target_link_libraries(gpujpegingest PUBLIC CUDA::cudart_static ${AV1GPU_NVJPEG} JPEG::JPEG Threads::Threads ${CMAKE_DL_LIBS}
  $<$<NOT:$<PLATFORM_ID:Windows>>:rt>)

# wavefront reconstruction, one instantiation (layout x bit depth) per file.  Their compile memory depends on the
# toolkit and the targets: peak RSS per file up to 18.4 GB with CUDA 12.8 (75;120, cicc, 10-bit), 10.4 GB with
# CUDA 13.2 for the default targets (ptxas for sm_100 / sm_120), under 2 GB with CUDA 12.9 / 13.0 or for one older target
# (L4 instance, 2026-10-31).  So with Ninja at most AV1GPU_RECON_JOBS of them compile at once (the rest of the build
# stays parallel).  Default: one per 20 GB of memory available at configure time, 1-4.
if(NOT DEFINED AV1GPU_RECON_JOBS)
  cmake_host_system_information(RESULT _av1gpu_mem_mib QUERY AVAILABLE_PHYSICAL_MEMORY)
  math(EXPR _av1gpu_recon_jobs "${_av1gpu_mem_mib} / 20480")
  if(_av1gpu_recon_jobs LESS 1)
    set(_av1gpu_recon_jobs 1)
  elseif(_av1gpu_recon_jobs GREATER 4)
    set(_av1gpu_recon_jobs 4)
  endif()
  set(AV1GPU_RECON_JOBS ${_av1gpu_recon_jobs} CACHE STRING "k_recon files compiled at once (up to ~18 GB of memory each)")
  message(STATUS "av1gpu: ${_av1gpu_mem_mib} MiB available, compiling ${AV1GPU_RECON_JOBS} k_recon file(s) at once")
endif()
set_property(GLOBAL APPEND PROPERTY JOB_POOLS av1gpu_recon=${AV1GPU_RECON_JOBS})
add_library(av1gpu_recon OBJECT
  encoder/src/k_recon.cu encoder/src/k_recon_444_8.cu encoder/src/k_recon_420_10.cu encoder/src/k_recon_444_10.cu)
set_property(TARGET av1gpu_recon PROPERTY JOB_POOL_COMPILE av1gpu_recon)
target_include_directories(av1gpu_recon PRIVATE encoder/src)
target_compile_options(av1gpu_recon PRIVATE ${AV1GPU_CUDA_OPTIONS})
target_link_libraries(av1gpu_recon PRIVATE av1f av1core_gpu gpujpegingest PNG::PNG ${AV1GPU_WEBP} Threads::Threads)

# encoder: analysis, partition DP, wavefront reconstruction, batching
add_library(av1gpu_encoder STATIC
  encoder/src/k_misc.cu encoder/src/k_an_a.cu encoder/src/k_an_b.cu encoder/src/k_an_c.cu encoder/src/k_an_d.cu
  encoder/src/k_an_uv.cu $<TARGET_OBJECTS:av1gpu_recon> encoder/src/k_lossless.cu encoder/src/pipeline.cpp
  encoder/src/imageio.cpp encoder/src/entropy_gpu.cpp encoder/src/entropy_dev.cu encoder/src/batch.cpp)
target_include_directories(av1gpu_encoder PUBLIC encoder/src)
target_compile_options(av1gpu_encoder PRIVATE
  ${AV1GPU_CUDA_OPTIONS}
  "$<$<COMPILE_LANGUAGE:CXX>:${AV1GPU_CXX_WARNINGS}>")
target_link_libraries(av1gpu_encoder PUBLIC av1f av1core_gpu gpujpegingest PNG::PNG ${AV1GPU_WEBP} Threads::Threads)

if(WIN32)
  # the manifest makes the narrow (char) Windows APIs use UTF-8, so non-ASCII file names work (Windows 10 1903+)
  add_executable(av1gpu encoder/apps/av1gpu.cpp encoder/apps/av1gpu.manifest)
  # nvjpeg64_<major>.dll is loaded on first use, so --help works without it (and without the NVIDIA driver)
  target_link_options(av1gpu PRIVATE /DELAYLOAD:nvjpeg64_${CUDAToolkit_VERSION_MAJOR}.dll)
  target_compile_definitions(av1gpu PRIVATE AV1GPU_NVJPEG_DLL="nvjpeg64_${CUDAToolkit_VERSION_MAJOR}.dll")
  target_link_libraries(av1gpu delayimp winmm)
else()
  add_executable(av1gpu encoder/apps/av1gpu.cpp)
endif()
target_link_libraries(av1gpu av1gpu_encoder)
install(TARGETS av1gpu RUNTIME DESTINATION bin)

if(AV1GPU_TESTS)
  foreach(t gpu_fuzz_test gpu_postfilter_test)
    add_executable(${t} tests/${t}.cu)
    target_link_libraries(${t} av1core_gpu)
    target_compile_options(${t} PRIVATE ${AV1GPU_CUDA_OPTIONS})
  endforeach()
  add_executable(gpu_dsp_test tests/gpu_dsp_test.cu)
  target_link_libraries(gpu_dsp_test av1core CUDA::cudart_static)
  target_compile_options(gpu_dsp_test PRIVATE ${AV1GPU_CUDA_OPTIONS})
  add_executable(varboost_test tests/varboost_test.cu)
  target_link_libraries(varboost_test av1f)
  target_compile_options(varboost_test PRIVATE ${AV1GPU_CUDA_OPTIONS})
  # decoder identity on tests/gate_images (dav1d + aomdec): cfl_edge_211x149.png gets CfL blocks with a split
  # luma tx on the bottom frame edge (CfL MaxLumaW/H); AB-only partitions make --preset slow reach them
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME gate_cfl_edge COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/decoder_gate.py
             --bin $<TARGET_FILE:av1gpu> --images ${PROJECT_SOURCE_DIR}/tests/gate_images --q 60,120
             -- --preset slow --small-preset none --partN 0xF0)
    # unreadable inputs (a directory, a missing or empty file) are skipped, the rest is encoded
    add_test(NAME batch_skip_unreadable COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/batch_skip_test.py
             --bin $<TARGET_FILE:av1gpu> --image ${PROJECT_SOURCE_DIR}/tests/gate_images/cfl_edge_211x149.png)
  endif()
endif()
