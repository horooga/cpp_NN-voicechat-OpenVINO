#include "audio_utils.hpp"
#include "cpp_NN-voicechat-OpenVINO/face_processing.hpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <openvino/genai/whisper_pipeline.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/core.hpp>
#include <thread>

std::ifstream idle0, idle_round, idle_open, idle_teeth;
std::ifstream smile0, smile1, smile_round, smile_open, smile_teeth;

bool mainMode = true;
std::atomic<bool> speak_flag = false;
std::atomic<bool> emotion_recognition_flag = false;
std::mutex mtx;
std::string shared_answer_string;
std::string shared_question_string;

void emotionRecognition() {
    try {
        ov::Core core;
        core.add_extension("libopenvino_tokenizers.so");
        auto tokenizer_model =
            core.read_model("./emotion_model_ov/openvino_tokenizer.xml",
                            "./emotion_model_ov/openvino_tokenizer.bin");
        auto compiled_tokenizer = core.compile_model(tokenizer_model, "GPU");
        auto tokenizer_infer_request =
            compiled_tokenizer.create_infer_request();

        ov::Tensor input_tensor(ov::element::string, {1});
        input_tensor.data<std::string>()[0] = shared_question_string;
        tokenizer_infer_request.set_input_tensor(input_tensor);
        tokenizer_infer_request.infer();
        auto input_ids_tensor = tokenizer_infer_request.get_tensor("input_ids");
        auto attention_mask_tensor =
            tokenizer_infer_request.get_tensor("attention_mask");

        std::shared_ptr<ov::Model> model =
            core.read_model("./emotion_model_ov/openvino_model.xml",
                            "./emotion_model_ov/openvino_model.bin");
        ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
        int input_ids_index, attention_mask_index;
        const auto &inputs = model->inputs();
        for (size_t i = 0; i < inputs.size(); i++) {
            std::string name = inputs[i].get_any_name();
            if (name == "input_ids") {
                input_ids_index = i;
            } else if (name == "attention_mask") {
                attention_mask_index = i;
            }
        }

        ov::InferRequest infer_request = compiled_model.create_infer_request();
        infer_request.set_input_tensor(input_ids_index, input_ids_tensor);
        infer_request.set_input_tensor(attention_mask_index,
                                       attention_mask_tensor);
        infer_request.infer();

        ov::Tensor output_tensor = infer_request.get_output_tensor();
        const float *output_data = output_tensor.data<const float>();
        size_t num_labels = output_tensor.get_shape()[1];

        size_t max_index = 0;
        float max_value = output_data[0];
        for (size_t i = 1; i < num_labels; i++) {
            if (output_data[i] > max_value) {
                max_value = output_data[i];
                max_index = i;
            }
        }

        std::map<size_t, std::string> id2label = {
            {0, "astonishment"}, {1, "angry"}, {2, "sad"}, {3, "smile"}};

        std::cout << "Predicted label: " << id2label[max_index]
                  << " (score: " << max_value << ")" << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Exception occurred: " << e.what() << std::endl;
        std::exit(1);
    }
}

void face_processing(const int &wpm) {
    std::string pwd = std::getenv("PWD");
    idle0 = std::ifstream("face-processing/idle0.txt");
    if (!idle0.is_open()) {
        std::cerr << "Error opening idle0.txt" << std::endl;
        std::exit(1);
    }
    idle_round = std::ifstream("face-processing/idle-round.txt");
    if (!idle_round.is_open()) {
        std::cerr << "Error opening idle-round.txt" << std::endl;
        std::exit(1);
    }
    idle_open = std::ifstream("face-processing/idle-open.txt");
    if (!idle_open.is_open()) {
        std::cerr << "Error opening idle-open.txt" << std::endl;
        std::exit(1);
    }
    idle_teeth = std::ifstream("face-processing/idle-teeth.txt");
    if (!idle_teeth.is_open()) {
        std::cerr << "Error opening idle-teeth.txt" << std::endl;
        std::exit(1);
    }
    smile0 = std::ifstream(pwd + "/face-processing/smile0.txt");
    if (!smile0.is_open()) {
        std::cerr << "Error opening smile0.txt" << std::endl;
        std::exit(1);
    }
    smile1 = std::ifstream(pwd + "/face-processing/smile1.txt");
    if (!smile1.is_open()) {
        std::cerr << "Error opening smile1.txt" << std::endl;
        std::exit(1);
    }
    smile_round = std::ifstream(pwd + "/face-processing/smile-round.txt");
    if (!smile_round.is_open()) {
        std::cerr << "Error opening smile-round.txt" << std::endl;
        std::exit(1);
    }
    smile_open = std::ifstream(pwd + "/face-processing/smile-open.txt");
    if (!smile_open.is_open()) {
        std::cerr << "Error opening smile-open.txt" << std::endl;
        std::exit(1);
    }
    smile_teeth = std::ifstream(pwd + "/face-processing/smile-teeth.txt");
    if (!smile_teeth.is_open()) {
        std::cerr << "Error opening smile-teeth.txt" << std::endl;
        std::exit(1);
    }

    std::string answer;
    std::chrono::milliseconds millisecsforword =
        std::chrono::milliseconds(60000 / wpm);
    while (true) {
        if (speak_flag.load()) {
            std::lock_guard<std::mutex> lock(mtx);
            answer = shared_answer_string;
        } else {
            continue;
        }
        std::istringstream answerSS(answer);
        std::string word;
        while (std::getline(answerSS, word, ' ')) {
            std::atomic<bool> stopFlag = false;
            std::thread wordProcessingThread(lipSync, word, std::ref(stopFlag));
            std::this_thread::sleep_for(millisecsforword);
            if (!wordProcessingThread.joinable()) {
                stopFlag.store(true);
            }
            wordProcessingThread.join();
        }
    };
}

void nnProcessing(const std::string &question) {
    ov::AnyMap whisperProperties = {{"PERFORMANCE_HINT", "LATENCY"},
                                    {"INFERENCE_PRECISION_HINT", "FP16"},
                                    {"CACHE_DIR", "./cache/WhisperPipeline"}};

    ov::genai::WhisperPipeline whisperPipeline(
        "./distil-whisper-large-v3-int4-ov", "GPU", whisperProperties);

    ov::AnyMap llamaProperties = {{"PERFORMANCE_HINT", "LATENCY"},
                                  {"INFERENCE_PRECISION_HINT", "FP16"},
                                  {"CACHE_DIR", "./cache/LlamaPipeline"}};
    ov::genai::LLMPipeline llamaPipeline("./tiny-llama-chat-ov", "GPU",
                                         llamaProperties);

    llamaPipeline.generate("You are just a collucutor in a casual "
                           "conversation. Keep it in mind and please "
                           "do not answer to this initial message");

    std::string recognizedText;
    do {
        if (mainMode) {
            auto rawAudio =
                utils::audio::record_audio(SAMPLE_RATE, CHANNELS, SECONDS);
            auto result = whisperPipeline.generate(rawAudio);
            std::stringstream ss;
            for (auto &s : result.texts) {
                ss << s;
            }
            recognizedText = ss.str();
        } else {
            recognizedText = question;
        }

        try {
            std::lock_guard<std::mutex> lock(mtx);
            shared_answer_string = llamaPipeline.generate(recognizedText);
        } catch (const std::exception &e) {
            std::cerr << "LLMPipeline exception: " << e.what() << std::endl;
            std::exit(1);
        }
        std::ofstream ofs("./answer.txt");
        {
            std::lock_guard<std::mutex> lock(mtx);
            ofs << shared_answer_string;
            ofs.close();
        }

        std::string cmd =
            "espeak-ng/build/src/espeak-ng -v en+belinda -p 70 -s 150 "
            " --stdout -f "
            "./answer.txt | paplay";
        std::cout << "used tts command: " << cmd << std::endl;
        speak_flag.store(true);
        int tts_res = std::system(cmd.c_str());
        speak_flag.store(false);
        if (tts_res != 0) {
            std::cerr << "Error during text-to-speech playback.\n";
            std::exit(1);
        }
    } while (mainMode);
    std::exit(0);
}

int main(int argc, char *argv[]) {
    std::string question;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--text-input") == 0) {
            if (i + 1 >= argc) {
                std::cerr << "Error: missing argument for --text-input\n";
                return 1;
            }
            mainMode = false;
            question = argv[i + 1];
            i++;
        }
    }

    std::thread emotion_recognition_thread(emotionRecognition);
    std::thread nn_processing_thread(nnProcessing, question);
    std::thread face_processing_thread(face_processing, 150);

    emotion_recognition_thread.join();
    nn_processing_thread.join();
    face_processing_thread.join();

    return 0;
}
