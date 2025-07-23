#include "audio_utils.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include <fstream>
#include <iostream>
#include <openvino/genai/whisper_pipeline.hpp>
#include <thread>

void write_text_to_file(const std::string &filename, const std::string &text) {
	std::ofstream ofs(filename);
	ofs << text;
	ofs.close();
}

int main(int argc, char *argv[]) {
	ov::genai::WhisperPipeline whisper_pipeline(
		"./distil-whisper-large-v3-int4-ov", "GPU");
	ov::genai::LLMPipeline llama_pipeline("./tiny-llama-chat-ov", "GPU");
	llama_pipeline.generate("You are a collucutor in a casual "
							"conversation. Keep it in mind and do not answer "
							"to this initial message");

	while (true) {
		auto raw_audio =
			utils::audio::record_audio(SAMPLE_RATE, CHANNELS, SECONDS);
		auto result = whisper_pipeline.generate(raw_audio);
		std::stringstream ss;
		for (auto &s : result.texts) {
			ss << s;
		}
		std::string recognized_text = ss.str();
		if (recognized_text.empty()) {
			std::cout << "No speech detected. Trying again...\n";
			continue;
		}

		std::cout << "\n\n\n\n\n\033[93mRECOGNIZED TEXT:\n" << recognized_text;
		std::string answer;
		try {
			answer = llama_pipeline.generate(recognized_text);
		} catch (const std::exception &e) {
			std::cerr << "OpenVINO GenAI LLM error: " << e.what() << std::endl;
			break;
		}
		write_text_to_file("./answer.txt", answer);

		std::cout << "\n\nANSWER:\n" << answer << "\n\n\n\n\n\033[0m";
		std::string cmd = "espeak-ng/build/src/espeak-ng -v en+f3 --stdout -f "
						  "./answer.txt | paplay";
		std::cout << "used tts command: " << cmd << std::endl;
		int tts_res = std::system(cmd.c_str());
		if (tts_res != 0) {
			std::cerr << "Error during text-to-speech playback.\n";
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(200));
	}

	return 0;
}
