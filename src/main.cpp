#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include <fstream>
#include <iostream>
#include <openvino/genai/whisper_pipeline.hpp>
#include <thread>

void writeTextToFile(const std::string &filename, const std::string &text) {
	std::ofstream ofs(filename);
	ofs << text;
	ofs.close();
}

int main(int argc, char *argv[]) {

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
	std::string recognizedText, answer;
	while (true) {
		auto rawAudio =
			utils::audio::record_audio(SAMPLE_RATE, CHANNELS, SECONDS);
		auto result = whisperPipeline.generate(rawAudio);
		std::stringstream ss;
		for (auto &s : result.texts) {
			ss << s;
		}
		recognizedText = ss.str();
		if (recognizedText.empty()) {
			std::cout << "No speech detected. Trying again...\n";
			// continue;
			std::exit(1);
		}

		std::cout << "\n\n\n\n\n\033[93mRECOGNIZED TEXT:\n" << recognizedText;

		try {
			answer = llamaPipeline.generate(recognizedText);
		} catch (const std::exception &e) {
			std::cerr << "OpenVINO GenAI LLM error: " << e.what() << std::endl;
			// break;
			std::exit(1);
		}
		writeTextToFile("./answer.txt", answer);

		std::cout << "\n\nANSWER:\n" << answer << "\n\n\n\n\n\033[0m";
		std::string cmd = "espeak-ng/build/src/espeak-ng -v en+f3 --stdout -f "
						  "./answer.txt | paplay";
		std::cout << "used tts command: " << cmd << std::endl;
		int tts_res = std::system(cmd.c_str());
		if (tts_res != 0) {
			std::cerr << "Error during text-to-speech playback.\n";
		}
	}

	return 0;
}
