<<<<<<< HEAD
cmake_minimum_required(VERSION 3.7)
=======
cmake_minimum_required(VERSION 3.2)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
project(tvm C CXX)

# Utility functions
include(cmake/util/Util.cmake)
include(cmake/util/FindCUDA.cmake)
include(cmake/util/FindVulkan.cmake)
include(cmake/util/FindLLVM.cmake)
include(cmake/util/FindROCM.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
tvm_option(USE_CUDA "Build with CUDA" OFF)
tvm_option(USE_OPENCL "Build with OpenCL" OFF)
tvm_option(USE_VULKAN "Build with Vulkan" OFF)
tvm_option(USE_OPENGL "Build with OpenGL" OFF)
tvm_option(USE_METAL "Build with Metal" OFF)
tvm_option(USE_ROCM "Build with ROCM" OFF)
tvm_option(ROCM_PATH "The path to rocm" /opt/rocm)
tvm_option(USE_RPC "Build with RPC" ON)
tvm_option(USE_LLVM "Build with LLVM, can be set to specific llvm-config path" OFF)
tvm_option(USE_STACKVM_RUNTIME "Include stackvm into the runtime" OFF)
tvm_option(USE_GRAPH_RUNTIME "Build with tiny graph runtime" ON)
tvm_option(USE_GRAPH_RUNTIME_DEBUG "Build with tiny graph runtime debug mode" OFF)
tvm_option(USE_SGX "Build with SGX" OFF)
tvm_option(USE_RTTI "Build with RTTI" ON)
tvm_option(USE_MSVC_MT "Build with MT" OFF)
tvm_option(INSTALL_DEV "Install compiler infrastructure" OFF)

<<<<<<< HEAD
tvm_option(USE_CUDNN "Build with cuDNN" OFF)

include_directories("include")
include_directories("HalideIR/src")
include_directories("dlpack/include")
=======
# Contrib library options
tvm_option(USE_BLAS "The blas library to be linked" none)
tvm_option(USE_MKL_PATH "MKL root path when use MKL blas" none)
tvm_option(USE_CUDNN "Build with cuDNN" OFF)
tvm_option(USE_CUBLAS "Build with cuBLAS" OFF)
tvm_option(USE_MIOPEN "Build with ROCM:MIOpen" OFF)
tvm_option(USE_ROCBLAS "Build with ROCM:RoCBLAS" OFF)
tvm_option(USE_SORT "Build with sort support" OFF)
tvm_option(USE_NNPACK "Build with nnpack support" OFF)
tvm_option(USE_RANDOM "Build with random support" OFF)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

# include directories
include_directories("include")
include_directories("3rdparty/dlpack/include")
include_directories("3rdparty/dmlc-core/include")
include_directories("3rdparty/compiler-rt")

# initial variables
set(TVM_LINKER_LIBS "")
set(TVM_RUNTIME_LINKER_LIBS "")

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-DHalide_SHARED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
<<<<<<< HEAD
  set(CMAKE_C_FLAGS "-O3 -Wall -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -std=c++11")
=======
  set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND
      CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 7.0)
    set(CMAKE_CXX_FLAGS "-faligned-new ${CMAKE_CXX_FLAGS}")
  endif()
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
endif(MSVC)

# add source group
FILE(GLOB_RECURSE GROUP_SOURCE "src/*.cc" "3rdparty/HalideIR/src/*.cpp" "nnvm/src/*.cc")
FILE(GLOB_RECURSE GROUP_INCLUDE "src/*.h" "include/*.h" "3rdparty/HalideIR/src/*.h"
                                "nnvm/src/*.h" "nnvm/include/*.h")
assign_source_group("Source" ${GROUP_SOURCE})
assign_source_group("Include" ${GROUP_INCLUDE})

# Source file lists
file(GLOB COMPILER_SRCS
    src/api/*.cc
    src/arithmetic/*.cc
    src/autotvm/*.cc
    src/codegen/*.cc
    src/lang/*.cc
    src/pass/*.cc
    src/op/*.cc
    src/schedule/*.cc
    )

file(GLOB_RECURSE RELAY_SRCS
    src/relay/*.cc
    )
list(APPEND COMPILER_SRCS ${RELAY_SRCS})


if(NOT MSVC)
  file(GLOB COMPILER_VERILOG_SRCS src/codegen/verilog/*.cc)
  list(APPEND COMPILER_SRCS ${COMPILER_VERILOG_SRCS})
endif()

file(GLOB_RECURSE NNVM_COMPILER_SRCS
    nnvm/src/c_api/*.cc
    nnvm/src/core/*.cc
    nnvm/src/pass/*.cc
    nnvm/src/compiler/*.cc
    nnvm/src/top/*.cc
    )

file(GLOB TOPI_SRCS
    topi/src/*.cc
)
<<<<<<< HEAD
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/topi/include)
file(GLOB TOPI_SRCS
    topi/src/*.cc
)
file(GLOB_RECURSE HALIDEIR_SRCS HalideIR/src/*.cpp)
list(APPEND COMPILER_SRCS ${HALIDEIR_SRCS})
file(GLOB RUNTIME_SRCS src/runtime/*.cc)
file(GLOB COMPILER_LLVM_SRCS src/codegen/llvm/*.cc)
file(GLOB COMPILER_VULKAN_SRCS src/codegen/spirv/*.cc)
file(GLOB RUNTIME_CUDA_SRCS src/runtime/cuda/*.cc)
file(GLOB RUNTIME_OPENCL_SRCS src/runtime/opencl/*.cc)
file(GLOB RUNTIME_OPENGL_SRCS src/runtime/opengl/*.cc)
file(GLOB RUNTIME_VULKAN_SRCS src/runtime/vulkan/*.cc)
file(GLOB RUNTIME_METAL_SRCS src/runtime/metal/*.mm)
file(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
file(GLOB RUNTIME_GRAPH_SRCS src/runtime/graph/*.cc)

if(USE_CUDA)
find_package(CUDA)
# Find CUDA doesn't find all the libraries we need, add the extra ones
find_library(CUDA_CUDA_LIBRARIES cuda
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
find_library(CUDA_NVRTC_LIBRARIES nvrtc
  PATHS ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib targets/x86_64-linux/lib/stubs)
  set(CUDA_CUDA_LIBRARY ${CUDA_CUDA_LIBRARIES})

  find_package(CUDA QUIET REQUIRED)
  message(STATUS "Build with CUDA support")
	include_directories(${CUDA_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDART_LIBRARY})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDA_LIBRARY})
  list(APPEND RUNTIME_SRCS ${RUNTIME_CUDA_SRCS})
  if(MSVC)
    find_library(CUDA_NVRTC_LIB nvrtc
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
    list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIB})
  else(MSVC)
    find_library(CUDA_NVRTC_LIB nvrtc
      ${CUDA_TOOLKIT_ROOT_DIR}/lib64
      ${CUDA_TOOLKIT_ROOT_DIR}/lib)
    list(APPEND TVM_LINKER_LIBS ${CUDA_NVRTC_LIB})
  endif(MSVC)

  if(USE_CUDNN)
    message(STATUS "Build with cuDNN support")
    file(GLOB CONTRIB_CUDNN_SRCS src/contrib/cudnn/*.cc)
    list(APPEND RUNTIME_SRCS ${CONTRIB_CUDNN_SRCS})
    if(MSVC)
      find_library(CUDA_CUDNN_LIB cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/win32)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIB})
    else(MSVC)
      find_library(CUDA_CUDNN_LIB cudnn
        ${CUDA_TOOLKIT_ROOT_DIR}/lib64
        ${CUDA_TOOLKIT_ROOT_DIR}/lib)
      list(APPEND TVM_RUNTIME_LINKER_LIBS ${CUDA_CUDNN_LIB})
    endif(MSVC)
  endif(USE_CUDNN)

  add_definitions(-DTVM_CUDA_RUNTIME=1)
else(USE_CUDA)
  add_definitions(-DTVM_CUDA_RUNTIME=0)
endif(USE_CUDA)

if(USE_OPENCL)
  find_package(OpenCL QUIET REQUIRED)
  message(STATUS "Build with OpenCL support")
  include_directories(${OpenCL_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenCL_LIBRARIES})
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENCL_SRCS})
  add_definitions(-DTVM_OPENCL_RUNTIME=1)
else(USE_OPENCL)
  add_definitions(-DTVM_OPENCL_RUNTIME=0)
endif(USE_OPENCL)

if(USE_OPENGL)
  find_package(OpenGL QUIET REQUIRED)
  find_package(glfw3 QUIET REQUIRED)
  message(STATUS "Build with OpenGL support")
  include_directories(${OPENGL_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${OpenGL_LIBRARIES} glfw)
  list(APPEND RUNTIME_SRCS ${RUNTIME_OPENGL_SRCS})
  add_definitions(-DTVM_OPENGL_RUNTIME=1)
else(USE_OPENGL)
  add_definitions(-DTVM_OPENGL_RUNTIME=0)
endif(USE_OPENGL)

if(USE_VULKAN)
  find_package(Vulkan REQUIRED)
  message(STATUS "Build with VULKAN support")
  include_directories(${Vulkan_INCLUDE_DIRS})
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${Vulkan_LIBRARIES})
  list(APPEND RUNTIME_SRCS ${RUNTIME_VULKAN_SRCS})
  list(APPEND COMPILER_SRCS ${COMPILER_VULKAN_SRCS})
  get_filename_component(VULKAN_LIB_PATH ${Vulkan_LIBRARY} DIRECTORY)
  find_library(SPIRV_TOOLS_LIB SPIRV-Tools
               ${VULKAN_LIB_PATH}/spirv-tools)
  list(APPEND TVM_LINKER_LIBS ${SPIRV_TOOLS_LIB})
  add_definitions(-DTVM_VULKAN_RUNTIME=1)
else(USE_VULKAN)
  add_definitions(-DTVM_VULKAN_RUNTIME=0)
endif(USE_VULKAN)

if(USE_METAL)
  find_package(OpenCL QUIET REQUIRED)
  message(STATUS "Build with Metal support")
  FIND_LIBRARY(METAL_LIB Metal)
  FIND_LIBRARY(FOUNDATION_LIB Foundation)
  list(APPEND TVM_RUNTIME_LINKER_LIBS ${METAL_LIB} ${FOUNDATION_LIB})
  list(APPEND RUNTIME_SRCS ${RUNTIME_METAL_SRCS})
  add_definitions(-DTVM_METAL_RUNTIME=1)
else(USE_METAL)
  add_definitions(-DTVM_METAL_RUNTIME=0)
endif(USE_METAL)
=======
file(GLOB_RECURSE HALIDEIR_SRCS 3rdparty/HalideIR/src/*.cpp)
list(APPEND COMPILER_SRCS ${HALIDEIR_SRCS})
file(GLOB RUNTIME_SRCS src/runtime/*.cc)

# Package runtime rules
if(NOT USE_RTTI)
  add_definitions(-DDMLC_ENABLE_RTTI=0)
endif()
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199

if(USE_RPC)
  message(STATUS "Build with RPC support...")
  file(GLOB RUNTIME_RPC_SRCS src/runtime/rpc/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_RPC_SRCS})
endif(USE_RPC)

file(GLOB STACKVM_RUNTIME_SRCS src/runtime/stackvm/*.cc)
file(GLOB STACKVM_CODEGEN_SRCS src/codegen/stackvm/*.cc)
list(APPEND COMPILER_SRCS ${STACKVM_CODEGEN_SRCS})
if(USE_STACKVM_RUNTIME)
  message(STATUS "Build with stackvm support in runtime...")
  list(APPEND RUNTIME_SRCS ${STACKVM_RUNTIME_SRCS})
else()
  list(APPEND COMPILER_SRCS ${STACKVM_RUNTIME_SRCS})
endif(USE_STACKVM_RUNTIME)

if(USE_GRAPH_RUNTIME)
  message(STATUS "Build with Graph runtime support...")
  file(GLOB RUNTIME_GRAPH_SRCS src/runtime/graph/*.cc)
  list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_SRCS})

  if(USE_GRAPH_RUNTIME_DEBUG)
    message(STATUS "Build with Graph runtime debug support...")
    file(GLOB RUNTIME_GRAPH_DEBUG_SRCS src/runtime/graph/debug/*.cc)
    list(APPEND RUNTIME_SRCS ${RUNTIME_GRAPH_DEBUG_SRCS})
    set_source_files_properties(${RUNTIME_GRAPH_SRCS}
      PROPERTIES COMPILE_DEFINITIONS "TVM_GRAPH_RUNTIME_DEBUG")
  endif(USE_GRAPH_RUNTIME_DEBUG)
endif(USE_GRAPH_RUNTIME)

# Module rules
include(cmake/modules/VTA.cmake)
include(cmake/modules/CUDA.cmake)
include(cmake/modules/OpenCL.cmake)
include(cmake/modules/OpenGL.cmake)
include(cmake/modules/Vulkan.cmake)
include(cmake/modules/Metal.cmake)
include(cmake/modules/ROCM.cmake)
include(cmake/modules/SGX.cmake)
include(cmake/modules/LLVM.cmake)
include(cmake/modules/contrib/BLAS.cmake)
include(cmake/modules/contrib/Random.cmake)
include(cmake/modules/contrib/Sort.cmake)
include(cmake/modules/contrib/NNPack.cmake)

add_library(tvm SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS})
add_library(tvm_topi SHARED ${TOPI_SRCS})
add_library(tvm_runtime SHARED ${RUNTIME_SRCS})
if(NOT USE_SGX STREQUAL "OFF")
  add_dependencies(tvm sgx_edl)
  add_dependencies(tvm_runtime sgx_edl tvm_t)
  install(TARGETS tvm_t ARCHIVE DESTINATION lib${LIB_SUFFIX})
endif()
add_library(nnvm_compiler SHARED ${NNVM_COMPILER_SRCS})

target_link_libraries(tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_topi tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_runtime ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(nnvm_compiler tvm)

# Related headers
target_include_directories(
  tvm
  PUBLIC "3rdparty/HalideIR/src"
  PUBLIC "topi/include")
target_include_directories(
  tvm_topi
  PUBLIC "topi/include")
target_include_directories(
  nnvm_compiler
  PUBLIC "nnvm/include"
  PUBLIC "topi/include")

# Tests
set(TEST_EXECS "")
file(GLOB TEST_SRCS tests/cpp/*.cc)
find_library(GTEST_LIB gtest "$ENV{GTEST_LIB}")

if(GTEST_LIB)
  foreach(__srcpath ${TEST_SRCS})
    get_filename_component(__srcname ${__srcpath} NAME)
    string(REPLACE ".cc" "" __execname ${__srcname})
    add_executable(${__execname} ${__srcpath})
    list(APPEND TEST_EXECS ${__execname})
    target_link_libraries(${__execname}
      tvm ${GTEST_LIB} pthread)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_ALL 1)
    set_target_properties(${__execname} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
  endforeach()
  add_custom_target(cpptest DEPENDS ${TEST_EXECS})
endif()

<<<<<<< HEAD
list(APPEND RUNTIME_SRCS ${GROUP_Include})
add_library(tvm SHARED ${COMPILER_SRCS} ${RUNTIME_SRCS})
add_library(tvm_topi SHARED ${TOPI_SRCS})
add_library(tvm_runtime SHARED ${RUNTIME_SRCS})
target_link_libraries(tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_topi tvm ${TVM_LINKER_LIBS} ${TVM_RUNTIME_LINKER_LIBS})
target_link_libraries(tvm_runtime  ${TVM_RUNTIME_LINKER_LIBS})
=======
# Custom targets
add_custom_target(runtime DEPENDS tvm_runtime)

# Installation rules
install(TARGETS tvm DESTINATION lib${LIB_SUFFIX})
install(TARGETS tvm_topi DESTINATION lib${LIB_SUFFIX})
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
install(TARGETS tvm_runtime DESTINATION lib${LIB_SUFFIX})
install(TARGETS nnvm_compiler DESTINATION lib${LIB_SUFFIX})

if (INSTALL_DEV)
  install(
    DIRECTORY "include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "topi/include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "3rdparty/HalideIR/src/." DESTINATION "include/HalideIR"
    FILES_MATCHING
    PATTERN "*.h"
  )
  install(
    DIRECTORY "3rdparty/dlpack/include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
    )
  install(
    DIRECTORY "nnvm/include/." DESTINATION "include"
    FILES_MATCHING
    PATTERN "*.h"
    )
else(INSTALL_DEV)
  install(
    DIRECTORY "include/tvm/runtime/." DESTINATION "include/tvm/runtime"
    FILES_MATCHING
    PATTERN "*.h"
    )
endif(INSTALL_DEV)

<<<<<<< HEAD
=======
# More target definitions
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
if(MSVC)
  target_compile_definitions(tvm PRIVATE -DHalide_EXPORTS)
  target_compile_definitions(tvm_runtime PRIVATE -DHalide_EXPORTS)
  target_compile_definitions(tvm PRIVATE -DTVM_EXPORTS)
  target_compile_definitions(tvm_runtime PRIVATE -DTVM_EXPORTS)
<<<<<<< HEAD
=======
  target_compile_definitions(nnvm_compiler PRIVATE -DNNVM_EXPORTS)
>>>>>>> 5e66870b31e16da7d0e95e5b0b4fc50d7cd02199
endif()
