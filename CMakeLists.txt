cmake_minimum_required(VERSION 3.20)
project(cpp_NN-voicechat-OpenVINO)

set(CMAKE_CXX_STANDARD 17)
set(TBB_ROOT "/opt/intel/openvino-genai_2025.2.0/runtime/3rdparty/tbb")

include_directories(
    /opt/intel/openvino_2025.2.0/runtime/include
    /opt/intel/openvino-genai_2025.2.0/runtime/include
    include
    ${TBB_ROOT}/include
)
link_directories(
    /opt/intel/openvino_2025.2.0/runtime/lib/intel64
    /opt/intel/openvino-genai_2025.2.0/runtime/lib/intel64
    ${TBB_ROOT}/lib
)

add_executable(cpp_NN-voicechat-OpenVINO src/main.cpp src/audio_utils.cpp src/face_processing.cpp)

target_link_libraries(cpp_NN-voicechat-OpenVINO
    openvino_tokenizers
    openvino_genai
    openvino
    portaudio
    tbb
)
