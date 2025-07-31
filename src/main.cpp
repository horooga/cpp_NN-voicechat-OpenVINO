#include "audio_utils.hpp"
#include "cpp_NN-voicechat-OpenVINO/face_processing.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include <fstream>
#include <iostream>
#include <mutex>
#include <openvino/genai/whisper_pipeline.hpp>
#include <thread>

std::mutex mtx;
std::string sharedAnswerString;

void faceProcessing() {
	std::string pwd = std::getenv("PWD");
	// std::ifstream idle0File("face-processing/idle0.txt");
	std::ifstream smile0File(pwd + "/face-processing/smile0.txt");
	std::ifstream smile1File(pwd + "/face-processing/smile1.txt");

	if (!smile0File.is_open()) {
		std::cout << "Error: smile0.txt file not found" << std::endl;
		return;
	}

	smile(smile0File, smile1File, mtx, sharedAnswerString);
}

void nnProcessing() {
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

	ov::AnyMap outeProperties = {{"PERFORMANCE_HINT", "LATENCY"},
								 {"INFERENCE_PRECISION_HINT", "FP16"},
								 {"CACHE_DIR", "./cache/OuteTts"}};

	llamaPipeline.generate("You are a collucutor in a casual "
						   "conversation. Keep it in mind and please "
						   "do not answer to this initial message");

	std::string recognizedText;
	while (true) {
		auto rawAudio =
			utils::audio::record_audio(SAMPLE_RATE, CHANNELS, SECONDS);
		auto result = whisperPipeline.generate(rawAudio);
		std::stringstream ss;
		for (auto &s : result.texts) {
			ss << s;
		}
		recognizedText = ss.str();

		try {
			std::lock_guard<std::mutex> lock(mtx);
			sharedAnswerString = llamaPipeline.generate(recognizedText);
		} catch (const std::exception &e) {
			break;
		}
		std::ofstream ofs("./answer.txt");
		{
			std::lock_guard<std::mutex> lock(mtx);
			ofs << sharedAnswerString;
			ofs.close();
		}

		std::string cmd = "espeak-ng/build/src/espeak-ng -v en+f3 --stdout -f "
						  "./answer.txt | paplay";
		std::cout << "used tts command: " << cmd << std::endl;
		int tts_res = std::system(cmd.c_str());
		if (tts_res != 0) {
			std::cerr << "Error during text-to-speech playback.\n";
		}
	}
}

int main(int argc, char *argv[]) {

	std::thread nnProcessingThread(nnProcessing);
	std::thread faceProcessingThread(faceProcessing);

	nnProcessingThread.join();
	faceProcessingThread.join();

	return 0;
}
