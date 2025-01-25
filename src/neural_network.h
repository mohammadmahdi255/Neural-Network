#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include <ap_fixed.h>
#include <cmath>
#include <cstdint>

#define OUTPUT_CHANNELS 8
#define KERNEL_SIZE 3
#define WIDTH 28
#define HEIGHT 28
#define MAX_POOL_SIZE 4

#define FLATTENED_SIZE                                                         \
  (WIDTH * HEIGHT * OUTPUT_CHANNELS) / (MAX_POOL_SIZE * MAX_POOL_SIZE)
#define HIDDEN_LAYER_128_SIZE 128
#define HIDDEN_LAYER_64_SIZE 64
#define OUTPUT_LAYER_SIZE 10

const int INT_BITS = 16; // Number of integer bits
const int FRAC_BITS = 8; // Number of fractional bits
typedef ap_fixed<INT_BITS + FRAC_BITS, INT_BITS> fixed_point_t;

enum class Command : uint8_t { IDLE = 0, CALCULATE = 1 };

struct Inputs {
  fixed_point_t data[HEIGHT][WIDTH];
  Command command;
};

struct Outputs {
  fixed_point_t data[HEIGHT][WIDTH][OUTPUT_CHANNELS];
};

const fixed_point_t conv2d_weights[KERNEL_SIZE][KERNEL_SIZE][OUTPUT_CHANNELS] =
    {{
         {0.44445667, -0.31208503, -0.38306206, -0.02557618, -0.21744953,
          0.12755907, -0.05988601, 0.31664094},
         {0.05663982, 0.12643431, -0.73759735, 0.5116596, -0.43275753,
          0.09629048, 0.06012313, 0.2635152},
         {-0.32628223, 0.32957625, -0.32836425, 0.19833377, -0.4594929,
          0.23173821, -0.20193106, -0.5274641},
     },
     {
         {-0.13916583, -0.5079763, -0.60725224, -0.5877137, -0.1765088,
          0.25832123, -0.6517595, 0.03957689},
         {0.41903448, -0.04111334, -0.0139517, 0.05099457, 0.01856276,
          -0.05466611, -0.36921304, 0.28115857},
         {0.15545268, 0.20245361, 0.58940315, 0.27543607, 0.37910694,
          0.11357204, -0.36583808, -0.46079773},
     },
     {
         {-0.32384333, 0.08540776, -0.516064, -0.2261248, 0.4444786, 0.23549715,
          0.5633444, 0.3743167},
         {-0.20887513, 0.24201189, -0.24331194, -0.6319109, 0.21686718,
          -0.0399669, 0.3991379, 0.18850237},
         {0.3257528, 0.22096887, 0.33261687, 0.09044164, 0.26522344, 0.17762055,
          -0.03445015, -0.01481458},
     }};

const fixed_point_t conv2d_biases[OUTPUT_CHANNELS] = {
    -0.33009830117225647,   -0.23910343647003174, -0.010327550582587719,
    -0.1464887410402298,    -0.12113799899816513, -0.4525485932826996,
    -0.0034316435921937227, -0.35708463191986084,
};

#endif
