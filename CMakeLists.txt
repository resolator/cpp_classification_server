cmake_minimum_required(VERSION 3.2)

project(server C CXX)

SET (SERVER_BINARY "server_binary")

find_package(OpenSSL)
find_package(Threads)
find_package(Poco REQUIRED COMPONENTS Foundation Util Net XML JSON Crypto NetSSL)

find_package(OpenCV REQUIRED)
find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime/core/session/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS /usr/local/lib)

if(NOT ${Poco_FOUND})
    message(FATAL_ERROR "Poco C++ Libraries not found.")
endif()

include_directories(${Poco_INCLUDE_DIRS})
link_directories("/usr/local/lib")

add_executable(${SERVER_BINARY} main.cpp)


target_include_directories(${SERVER_BINARY} PRIVATE "${CMAKE_BINARY_DIR}" "${ONNX_RUNTIME_SESSION_INCLUDE_DIRS}" "${OpenCV_INCLUDE_DIRS}")
target_compile_options(${SERVER_BINARY} PRIVATE -Wall -Wextra )


target_link_libraries(${SERVER_BINARY} PRIVATE
                             ${CMAKE_THREAD_LIBS_INIT} 
                             ${Poco_LIBRARIES}
                             "PocoData"
                             ${ONNX_RUNTIME_LIB}
                             ${OpenCV_LIBRARIES})

set_target_properties(${SERVER_BINARY} PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(${SERVER_BINARY} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)

install(TARGETS ${SERVER_BINARY} RUNTIME DESTINATION bin)

set(CPACK_GENERATOR DEB)
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_CONTACT v.dusiak@gmail.com)
include(CPack)
