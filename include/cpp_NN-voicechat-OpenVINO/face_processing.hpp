#include <fstream>
#include <mutex>

void processing();

void smile(std::ifstream &smile0, std::ifstream &smile1, std::mutex &mtx,
		   std::string &text);
