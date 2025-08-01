#include <atomic>
#include <chrono>
#include <fstream>

extern std::ifstream idle0, idle_round, idle_open, idle_teeth;
extern std::ifstream smile0, smile1, smile_round, smile_open, smile_teeth;

void lipSync(const std::string &text, std::atomic<bool> &stopFlag);
