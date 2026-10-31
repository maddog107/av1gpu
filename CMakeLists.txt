# av1gpu: a CUDA AV1 still-image (AVIF) encoder.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=native]
#   cmake --build build
#
# AV1GPU_CUDA=OFF builds only the CPU parts (bitstream writer, AVIF muxer and the decoder fuzz test),
# which need no CUDA toolkit.
cmake_minimum_required(VERSION 3.24)

# Default GPU targets (set after enable_language(CUDA), when the CUDA version is known): Turing through
# Blackwell.  Building for one architecture is much faster: -DCMAKE_CUDA_ARCHITECTURES=native (or e.g. 89).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(AV1GPU_DEFAULT_ARCHS ON)
endif()

project(av1gpu VERSION 0.1.0 LANGUAGES C CXX)

option(AV1GPU_CUDA "Build the CUDA encoder" ON)
option(AV1GPU_TESTS "Build the tests" ON)
# release binaries: libjpeg, libpng, zlib, libwebp and nvJPEG linked in, so only the NVIDIA driver is needed
option(AV1GPU_STATIC_DEPS "Link the image libraries and nvJPEG statically" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -g")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---------------------------------------------------------------------------------------------------
# av1core: decision record -> AV1 bitstream (CPU writer), AVIF muxer, shared DSP headers
add_library(av1core STATIC av1core/src/encoder.cpp av1core/src/avif.cpp)
target_include_directories(av1core PUBLIC av1core/include av1core/src)
target_compile_options(av1core PRIVATE -Wall -Wno-missing-braces -Wno-unused-variable -Wno-unused-function)

if(AV1GPU_TESTS)
  enable_testing()
  # random legal decision records -> av1core -> dav1d + aomdec must reproduce the reconstruction exactly
  add_executable(av1g_fuzz tests/fuzz.cpp)
  target_link_libraries(av1g_fuzz av1core)
  add_test(NAME fuzz_dav1d_aomdec COMMAND av1g_fuzz --seed 1 --count 200 --quiet)
endif()

if(NOT AV1GPU_CUDA)
  return()
endif()

# ---------------------------------------------------------------------------------------------------
enable_language(CUDA)
if(AV1GPU_DEFAULT_ARCHS)
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8)
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;100-real;120-real;120-virtual")
  else()
    set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-real;90-virtual")
  endif()
endif()
message(STATUS "CUDA ${CMAKE_CUDA_COMPILER_VERSION}, architectures: ${CMAKE_CUDA_ARCHITECTURES}")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -lineinfo")
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
find_package(CUDAToolkit 12.0 REQUIRED)
if(AV1GPU_STATIC_DEPS)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
  set(AV1GPU_NVJPEG CUDA::nvjpeg_static)
else()
  set(AV1GPU_NVJPEG CUDA::nvjpeg)
endif()
find_package(JPEG REQUIRED)
find_package(PNG REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(WEBP REQUIRED IMPORTED_TARGET libwebp)
find_package(Threads REQUIRED)

# av1core GPU backend: symbolization + entropy coding, deblocking + CDEF
add_library(av1core_gpu STATIC av1core/src/gpu_entropy.cu av1core/src/gpu_postfilter.cu)
target_link_libraries(av1core_gpu PUBLIC av1core CUDA::cudart_static)
# per-thread default stream: the entropy thread's batches run concurrently with the next encode
target_compile_options(av1core_gpu PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --default-stream per-thread>)

# av1f: libaom tune=iq decisions, variance boost, loop-filter / CDEF strength search
add_library(av1f STATIC av1f/src/varboost.cu av1f/src/lfsearch.cu)
target_include_directories(av1f PUBLIC av1f/include)
target_link_libraries(av1f PUBLIC av1core CUDA::cudart_static)
target_compile_options(av1f PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

# ingest: JPEG -> GPU-resident 4:2:0 planes (nvJPEG hardware/hybrid decode or CPU entropy decode + GPU IDCT)
add_library(gpujpegingest STATIC ingest/src/jpeg_reader.cpp ingest/src/kernels.cu ingest/src/pipeline.cu)
target_include_directories(gpujpegingest PUBLIC ingest/src)
target_link_libraries(gpujpegingest PUBLIC CUDA::cudart_static ${AV1GPU_NVJPEG} JPEG::JPEG Threads::Threads ${CMAKE_DL_LIBS} rt)

# encoder: analysis, partition DP, wavefront reconstruction, batching
add_library(av1gpu_encoder STATIC
  encoder/src/k_misc.cu encoder/src/k_an_a.cu encoder/src/k_an_b.cu encoder/src/k_an_c.cu encoder/src/k_an_d.cu
  encoder/src/k_an_uv.cu encoder/src/k_recon.cu encoder/src/pipeline.cpp encoder/src/imageio.cpp
  encoder/src/entropy_gpu.cpp encoder/src/entropy_dev.cu encoder/src/batch.cpp)
target_include_directories(av1gpu_encoder PUBLIC encoder/src)
target_compile_options(av1gpu_encoder PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wno-unused-function -Wno-missing-braces>)
target_link_libraries(av1gpu_encoder PUBLIC av1f av1core_gpu gpujpegingest PNG::PNG PkgConfig::WEBP Threads::Threads)

add_executable(av1gpu encoder/apps/av1gpu.cpp)
target_link_libraries(av1gpu av1gpu_encoder)
install(TARGETS av1gpu RUNTIME DESTINATION bin)

if(AV1GPU_TESTS)
  foreach(t gpu_fuzz_test gpu_postfilter_test)
    add_executable(${t} tests/${t}.cu)
    target_link_libraries(${t} av1core_gpu)
    target_compile_options(${t} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  endforeach()
  add_executable(gpu_dsp_test tests/gpu_dsp_test.cu)
  target_link_libraries(gpu_dsp_test av1core CUDA::cudart_static)
  target_compile_options(gpu_dsp_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  add_executable(varboost_test tests/varboost_test.cu)
  target_link_libraries(varboost_test av1f)
  target_compile_options(varboost_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  # decoder identity on tests/gate_images (dav1d + aomdec): cfl_edge_211x149.png gets CfL blocks with a split
  # luma tx on the bottom frame edge (CfL MaxLumaW/H); AB-only partitions make --preset slow reach them
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    add_test(NAME gate_cfl_edge COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/decoder_gate.py
             --bin $<TARGET_FILE:av1gpu> --images ${PROJECT_SOURCE_DIR}/tests/gate_images --q 60,120
             -- --preset slow --small-preset none --partN 0xF0)
  endif()
endif()
