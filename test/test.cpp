#include "../src/neural_network.h"
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

extern void controller(fixed_point_t inputs[HEIGHT][WIDTH], Command command, fixed_point_t outputs[OUTPUT_LAYER_SIZE]);

bool read_pgm(const std::string &filename, uint8_t *&image_matrix,
              uint16_t &height, uint16_t &width) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening PGM file." << std::endl;
    return false;
  }

  std::string line;
  std::getline(file, line);
  if (line != "P5") {
    std::cerr << "Invalid PGM format, expecting P5." << std::endl;
    return false;
  }

  do {
    std::getline(file, line);
  } while (line[0] == '#');

  std::istringstream dimensions(line);
  dimensions >> width >> height;

  image_matrix = new uint8_t[height * width];

  int max_pixel_value;
  file >> max_pixel_value;
  file.get();

  file.read(reinterpret_cast<char *>(image_matrix), height * width);

  file.close();
  return true;
}

void save_pgm(const std::string &filename, uint8_t *image_matrix,
              uint16_t height, uint16_t width) {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening output PGM file." << std::endl;
    return;
  }

  file << "P5\n";
  file << width << " " << height << "\n";
  file << "255\n";

  file.write(reinterpret_cast<char *>(image_matrix), width * height);

  file.close();
}

int main() {
  uint16_t width;
  uint16_t heigth;
  uint8_t* data;

  if (!read_pgm("../../../test/img-9.pgm", data, heigth, width)) {
    std::cout << "Could not load image" << std::endl;
    return 1;
  }

  fixed_point_t inputs[HEIGHT][WIDTH];
  fixed_point_t outputs[OUTPUT_LAYER_SIZE];

  for (int i = 0; i < heigth * width; ++i) {
	  inputs[i / width][i % width] = float(data[i]) / 255.0;
  }

  std::cout << "=========================================================" << std::endl;

  controller(inputs, Command::CALCULATE, outputs);

  for (int i = 0; i < OUTPUT_LAYER_SIZE; ++i) {
	  std::cout << outputs[i] << ", ";
  }

  std::cout << std::endl << "=========================================================" << std::endl;


  return 0;
}
