#include "neural_network.h"
#include "hidden_layer_128.h"
#include "hidden_layer_64.h"
#include "output_layer.h"

fixed_point_t apply_kernel(uint8_t row, uint8_t col,
                           const fixed_point_t input[HEIGHT][WIDTH],
                           uint8_t out_ch) {
#pragma HLS INLINE
  fixed_point_t sum = conv2d_biases[out_ch];

  for (int k = 0; k < KERNEL_SIZE * KERNEL_SIZE; k++) {
#pragma HLS UNROLL
    uint8_t kr = k / KERNEL_SIZE;
    uint8_t kc = k % KERNEL_SIZE;

    uint8_t in_row = row + kr - KERNEL_SIZE / 2;
    uint8_t in_col = col + kc - KERNEL_SIZE / 2;

    sum += input[in_row][in_col] * conv2d_weights[kr][kc][out_ch] *
           (in_row >= 0 && in_row < HEIGHT && in_col >= 0 && in_col < WIDTH);
  }

  return sum;
}

void conv2d(const fixed_point_t input[HEIGHT][WIDTH],
            fixed_point_t output[HEIGHT][WIDTH][OUTPUT_CHANNELS]) {

  for (uint8_t out_ch = 0; out_ch < OUTPUT_CHANNELS; out_ch++) {
#pragma HLS UNROLL
    for (uint8_t row = 0; row < HEIGHT; row++) {
#pragma HLS UNROLL
      for (uint8_t col = 0; col < WIDTH; col++) {
        fixed_point_t result = apply_kernel(row, col, input, out_ch);
        output[row][col][out_ch] = result * (result > 0);
      }
    }
  }
}

void max_poo_layer(const fixed_point_t input[HEIGHT][WIDTH][OUTPUT_CHANNELS],
                   fixed_point_t output[FLATTENED_SIZE]) {
  for (uint8_t out_ch = 0; out_ch < OUTPUT_CHANNELS; out_ch++) {
#pragma HLS UNROLL
    for (uint8_t row = 0; row < HEIGHT; row += MAX_POOL_SIZE) {
#pragma HLS UNROLL
      for (uint8_t col = 0; col < WIDTH; col += MAX_POOL_SIZE) {
        uint16_t idx =
            row * WIDTH * OUTPUT_CHANNELS / (MAX_POOL_SIZE * MAX_POOL_SIZE) +
            col * OUTPUT_CHANNELS / MAX_POOL_SIZE + out_ch;

        fixed_point_t max = input[row][col][out_ch];

        for (int k = 0; k < MAX_POOL_SIZE * MAX_POOL_SIZE; k++) {
#pragma HLS UNROLL
          uint8_t kr = k / MAX_POOL_SIZE;
          uint8_t kc = k % MAX_POOL_SIZE;

          uint8_t in_row = row + kr;
          uint8_t in_col = col + kc;

          if (max < input[in_row][in_col][out_ch])
            max = input[in_row][in_col][out_ch];
        }

        output[idx] = max;
      }
    }
  }
}

void hidden_layer_128(const fixed_point_t input[FLATTENED_SIZE],
                      fixed_point_t output[HIDDEN_LAYER_128_SIZE]) {
  for (int i = 0; i < HIDDEN_LAYER_128_SIZE; ++i) {
#pragma HLS UNROLL
    output[i] = hidden_biases_128[i];
    for (int j = 0; j < FLATTENED_SIZE; ++j) {
      output[i] += input[j] * hidden_weights_128[j][i];
    }
    if (output[i].to_float() < 0)
      output[i] = 0;
  }
}

void hidden_layer_64(const fixed_point_t input[HIDDEN_LAYER_128_SIZE],
                     fixed_point_t output[HIDDEN_LAYER_64_SIZE]) {
  for (int i = 0; i < HIDDEN_LAYER_64_SIZE; ++i) {
#pragma HLS UNROLL
    output[i] = hidden_biases_64[i];
    for (int j = 0; j < HIDDEN_LAYER_128_SIZE; ++j) {
      output[i] += input[j] * hidden_weights_64[j][i];
    }
    if (output[i].to_float() < 0)
      output[i] = 0;
  }
}

void output_layer(const fixed_point_t input[HIDDEN_LAYER_64_SIZE],
                  fixed_point_t output[OUTPUT_LAYER_SIZE]) {
  for (int i = 0; i < OUTPUT_LAYER_SIZE; ++i) {
#pragma HLS UNROLL
    output[i] = output_biases[i];
    for (int j = 0; j < HIDDEN_LAYER_64_SIZE; ++j) {
      output[i] += input[j] * output_weights[j][i];
    }
  }

  fixed_point_t exp_values[OUTPUT_LAYER_SIZE];
  fixed_point_t sum_exp = 0.0;

  for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
#pragma HLS UNROLL
    exp_values[i] = std::exp(output[i].to_float());
    sum_exp += exp_values[i];
  }

  for (int i = 0; i < OUTPUT_LAYER_SIZE; i++) {
#pragma HLS UNROLL
    output[i] = exp_values[i] / sum_exp;
  }
}

void controller(fixed_point_t inputs[HEIGHT][WIDTH], Command command,
                fixed_point_t outputs[OUTPUT_LAYER_SIZE]) {
#pragma HLS INTERFACE mode = ap_memory port = inputs
#pragma HLS INTERFACE mode = ap_memory port = outputs
#pragma HLS INTERFACE mode = ap_none port = command
#pragma HLS INTERFACE mode = ap_ctrl_none port = return

#pragma HLS ARRAY_PARTITION variable = inputs complete dim = 2

  fixed_point_t cov2d_out[HEIGHT][WIDTH][OUTPUT_CHANNELS];
  fixed_point_t max_pool_out[FLATTENED_SIZE];
  fixed_point_t hidden_layer_128_out[HIDDEN_LAYER_128_SIZE];
  fixed_point_t hidden_layer_64_out[HIDDEN_LAYER_64_SIZE];

  switch (command) {
  case Command::IDLE:
    // Do nothing
    break;
  case Command::CALCULATE:
    conv2d(inputs, cov2d_out);
    max_poo_layer(cov2d_out, max_pool_out);
    hidden_layer_128(max_pool_out, hidden_layer_128_out);
    hidden_layer_64(hidden_layer_128_out, hidden_layer_64_out);
    output_layer(hidden_layer_64_out, outputs);
    break;
  default:
    break;
  }
}
