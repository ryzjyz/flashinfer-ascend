cmake_minimum_required(VERSION 3.16)
include("${FI_RECIPE_FILE}")
find_package(ASC REQUIRED)
project(flashinfer_aicpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

include("${FI_OPERATOR_CMAKE}")
if(NOT EXISTS "${FI_AICPU_CONFIG}" OR NOT IS_DIRECTORY "${FI_ACLNN_HEADERS}")
  message(FATAL_ERROR "AICPU operators must set FI_AICPU_CONFIG and FI_ACLNN_HEADERS")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${FI_AICPU_CONFIG}")

execute_process(
  COMMAND "${ASCEND_PYTHON_EXECUTABLE}" -B -c [=[
import json
import re
import sys
from pathlib import Path

registration = json.loads(Path(sys.argv[1]).read_text())
op_type = sys.argv[2]
if set(registration) != {op_type}:
    raise ValueError("AICPU registry must identify exactly " + op_type)
library = registration[op_type]["opInfo"]["kernelSo"]
if not isinstance(library, str) or not re.fullmatch(r"lib[A-Za-z0-9_]+\.so", library):
    raise ValueError("invalid AICPU kernelSo name")
print(library)
]=] "${FI_AICPU_CONFIG}" "${FI_PACKAGE_OP_TYPE}"
  RESULT_VARIABLE config_status OUTPUT_VARIABLE aicpu_library_name
  ERROR_VARIABLE config_stderr OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT config_status EQUAL 0)
  message(FATAL_ERROR "Invalid AICPU registration: ${config_stderr}")
endif()

set(cann "${ASCEND_CANN_PACKAGE_PATH}")
set(sdk_roots "${cann}" "${cann}/ops_base" "${cann}/${CMAKE_SYSTEM_PROCESSOR}-linux")
set(host_includes)
set(aicpu_includes)
set(lib_dirs)
foreach(root IN LISTS sdk_roots)
  list(APPEND host_includes
    "${root}/include" "${root}/include/aclnn" "${root}/pkg_inc" "${root}/include/op_common"
    "${root}/pkg_inc/op_common" "${root}/include/op_common/op_host")
  list(APPEND lib_dirs "${root}/lib64" "${root}/lib")
  foreach(prefix IN ITEMS "${root}/include" "${root}/pkg_inc")
    list(APPEND aicpu_includes "${prefix}" "${prefix}/aicpu" "${prefix}/base"
      "${prefix}/aicpu_common/context" "${prefix}/aicpu_common/context/common"
      "${prefix}/aicpu_common/context/cpu_proto" "${prefix}/aicpu_common/context/utils"
      "${prefix}/experiment" "${prefix}/experiment/cce" "${prefix}/experiment/msprof"
      "${prefix}/experiment/datagw/aicpu/common")
  endforeach()
endforeach()

find_library(nnopbase NAMES nnopbase PATHS ${lib_dirs} NO_DEFAULT_PATH)
find_program(aicpu_cxx NAMES aarch64-target-linux-gnu-g++
  PATHS "${cann}/toolkit/toolchain/hcc/bin" NO_DEFAULT_PATH)
find_file(aicpu_context NAMES libaicpu_context.a PATHS ${lib_dirs} NO_DEFAULT_PATH)
find_file(aicpu_protobuf NAMES libbase_ascend_protobuf.a PATHS ${lib_dirs} NO_DEFAULT_PATH)
if(NOT nnopbase OR NOT aicpu_cxx OR NOT aicpu_context OR NOT aicpu_protobuf)
  message(FATAL_ERROR "AICPU builds require CANN nnopbase, the AICPU compiler, context and protobuf libraries")
endif()

set(typed_adapter "${CMAKE_BINARY_DIR}/fi_aclnn_adapter.cpp")
get_filename_component(adapter_generator_dir "${FI_ACLNN_GENERATOR}" DIRECTORY)
execute_process(
  COMMAND "${ASCEND_PYTHON_EXECUTABLE}" -B -c
    "import sys; sys.path.insert(0, sys.argv[1]); from aclnn_adapter import generate; generate(*sys.argv[2:])"
    "${adapter_generator_dir}" "${FI_ACLNN_RECIPE_FILE}" "${FI_ACLNN_HEADERS}" "${typed_adapter}"
  RESULT_VARIABLE adapter_status ERROR_VARIABLE adapter_stderr)
if(NOT adapter_status EQUAL 0)
  message(FATAL_ERROR "ACLNN NativeBinding mismatch: ${adapter_stderr}")
endif()

set(package_name "${FI_PACKAGE_VENDOR}")
npu_op_package(${package_name} TYPE RUN CONFIG INSTALL_PATH ${CMAKE_BINARY_DIR})
npu_op_library(cust_opapi ACLNN ${FI_HOST_SOURCES} "${typed_adapter}")
target_include_directories(cust_opapi PRIVATE "${FI_ACLNN_HEADERS}" ${FI_INCLUDES} ${host_includes})
target_compile_definitions(cust_opapi PRIVATE ${FI_DEFINITIONS})
target_compile_options(cust_opapi PRIVATE -fvisibility=hidden)
target_link_options(cust_opapi PRIVATE ${FI_LINK_OPTIONS})
target_link_libraries(cust_opapi PRIVATE "${nnopbase}")

set(device_options ${FI_COMPILE_OPTIONS})
foreach(definition IN LISTS FI_DEFINITIONS)
  list(APPEND device_options "-D${definition}")
endforeach()
foreach(directory IN LISTS FI_INCLUDES aicpu_includes host_includes)
  list(APPEND device_options "-I${directory}")
endforeach()
set(aicpu_library "${CMAKE_BINARY_DIR}/${aicpu_library_name}")
add_custom_command(OUTPUT "${aicpu_library}"
  COMMAND "${aicpu_cxx}" -std=c++17 -O2 -fPIC -shared -fvisibility=hidden
    -D_GLIBCXX_USE_CXX11_ABI=1 -DDISABLE_COMPILE_V1 -Dgoogle=ascend_private
    ${device_options} ${FI_SOURCES}
    -Wl,--whole-archive "${aicpu_context}" "${aicpu_protobuf}"
    -Wl,--no-whole-archive -Wl,-Bsymbolic -Wl,--exclude-libs=libbase_ascend_protobuf.a
    -o "${aicpu_library}"
  DEPENDS ${FI_SOURCES} ${FI_HEADERS} "${aicpu_context}" "${aicpu_protobuf}"
  VERBATIM)
add_custom_target(flashinfer_aicpu_kernel ALL DEPENDS "${aicpu_library}")
add_dependencies(cust_opapi flashinfer_aicpu_kernel)

install(FILES "${aicpu_library}"
  DESTINATION "packages/vendors/${package_name}/op_impl/cpu/aicpu_kernel/impl")
install(FILES "${FI_AICPU_CONFIG}" RENAME cust_aicpu_kernel.json
  DESTINATION "packages/vendors/${package_name}/op_impl/cpu/config")
npu_op_package_add(${package_name} LIBRARY cust_opapi)
