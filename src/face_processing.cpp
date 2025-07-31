#include "cpp_NN-voicechat-OpenVINO/face_processing.hpp"
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>

void clearScreen() { std::cout << "\033[2J\033[1;1H"; }

void printFace(std::ifstream &file) {
	std::string line;
	while (std::getline(file, line)) {
		std::cout << line << std::endl;
	}
	file.clear();
	file.seekg(0, std::ios::beg);
}

void printSubtitle(std::string &text) {
	std::cout << "============================================================="
				 "============"
			  << std::endl;
	std::cout << text << std::endl;
	std::cout << "============================================================="
				 "============"
			  << std::endl;
}

void smile(std::ifstream &file0, std::ifstream &file1, std::mutex &mtx,
		   std::string &sharedText) {
	while (true) {
		{
			std::lock_guard<std::mutex> lock(mtx);
			printFace(file0);
			printSubtitle(sharedText);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		clearScreen();
		{
			std::lock_guard<std::mutex> lock(mtx);
			printFace(file1);
			printSubtitle(sharedText);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		clearScreen();
		{
			std::lock_guard<std::mutex> lock(mtx);
			printFace(file0);
			printSubtitle(sharedText);
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		clearScreen();
	}
}
