cmake_minimum_required(VERSION 3.20)
project(cpp_NN-voicechat-OpenVINO)

set(CMAKE_CXX_STANDARD 17)

include_directories(
    /opt/intel/openvino_2025.2.0/runtime/include
    /opt/intel/openvino_genai_2025.2.0/runtime/include
    include
)
link_directories(
    /opt/intel/openvino_2025.2.0/runtime/lib/intel64
    /opt/intel/openvino_genai_2025.2.0/runtime/lib/intel64
)

add_executable(cpp_NN-voicechat-OpenVINO src/main.cpp src/audio_utils.cpp)

target_link_libraries(cpp_NN-voicechat-OpenVINO
    openvino_genai
    openvino
    portaudio
)
