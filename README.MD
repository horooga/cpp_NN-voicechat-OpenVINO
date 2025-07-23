App for neural net voice chat using openvino
(espeak-ng for tts used cause perfomance purposes)

    (upper branch is only planned)
                    -emotion recognition-> recognized emotion -face gen algorithm-> image output
                  /
    audio input -|
                  \
                    -whisper-> recognized text -tiny llama chat-> saved ./answer.txt -espeak-ng via stdout to audio player-> audio output

**Requirements:**

- openvino runtime 2025.2.0 (natively installed)
- openvino genai 2025.2.0 (natively installed)
- portaudio

**Installing:**

- `pip install -r ./requirements.txt`
- `python ./install_stt_ttt.py`
- `sh install_tts.sh`

**Build:**

```bash
source /opt/intel/openvino_2025.2.0/setupvars.sh
source /opt/intel/openvino_genai_2025.2.0/setupvars.sh
mkdir build
cmake -S . -B build
cmake --build build
```

**Run:**
Recommended to use from project root dir cause of relative paths

```bash
build/cpp_NN-voicechat-OpenVINO
```
