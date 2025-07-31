git clone https://github.com/espeak-ng/espeak-ng
cd espeak-ng
mkdir build
cmake -S . -B build
cmake --build build
cd build
make
