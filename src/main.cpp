#include "audio_utils.hpp"
#include "cpp_NN-voicechat-OpenVINO/face_processing.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <openvino/genai/whisper_pipeline.hpp>
#include <thread>

std::ifstream idle0, idle_round, idle_open, idle_teeth;
std::ifstream smile0, smile1, smile_round, smile_open, smile_teeth;

bool mainMode = true;
std::atomic<bool> speakFlag = false;
std::mutex mtx;
std::string sharedAnswerString;

void faceProcessing(const int &wpm) {
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
		if (speakFlag.load()) {
			std::lock_guard<std::mutex> lock(mtx);
			answer = sharedAnswerString;
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
			sharedAnswerString = llamaPipeline.generate(recognizedText);
		} catch (const std::exception &e) {
			std::cerr << "LLMPipeline exception: " << e.what() << std::endl;
			std::exit(1);
		}
		std::ofstream ofs("./answer.txt");
		{
			std::lock_guard<std::mutex> lock(mtx);
			ofs << sharedAnswerString;
			ofs.close();
		}

		std::string cmd =
			"espeak-ng/build/src/espeak-ng -v en+belinda -p 70 -s 150 "
			" --stdout -f "
			"./answer.txt | paplay";
		std::cout << "used tts command: " << cmd << std::endl;
		speakFlag.store(true);
		int tts_res = std::system(cmd.c_str());
		speakFlag.store(false);
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

	std::thread nnProcessingThread(nnProcessing, question);
	std::thread faceProcessingThread(faceProcessing, 150);

	nnProcessingThread.join();
	faceProcessingThread.join();

	return 0;
}
