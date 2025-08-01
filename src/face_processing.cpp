#include "cpp_NN-voicechat-OpenVINO/face_processing.hpp"
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>

const int millisecsforletter = 69;

void clearScreen() { std::cout << "\033[2J\033[1;1H"; }

void printFace(std::ifstream &file) {
	std::string line;
	while (std::getline(file, line)) {
		std::cout << line << std::endl;
	}
	file.clear();
	file.seekg(0, std::ios::beg);
}

void lipSync(const std::string &word, std::atomic<bool> &stopFlag) {
	for (auto c : word) {
		clearScreen();

		if (stopFlag.load()) {
			std::cout << "Time for one word is over" << std::endl;
			return;
		}
		switch (c) {
		case 'a':
		case 'i':
		case 'o':
		case 'u':
			printFace(idle_open);
			break;
		case 'b':
		case 'c':
		case 'd':
		case 'e':
		case 'f':
		case 'g':
		case 'y':
		case 'j':
		case 'p':
		case 's':
		case 'x':
			printFace(idle_teeth);
			break;
		case 'r':
		case 'w':
			printFace(idle_round);
			break;
		default:
			printFace(idle0);
			break;
		}
		std::this_thread::sleep_for(
			std::chrono::milliseconds(millisecsforletter));
	}
}
