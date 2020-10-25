#ifndef CNN_PROPAGATION_H
#define CNN_PROPAGATION_H

#include "stdint.h"

#include <vector>
using std::vector;

void prop_forward(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);

void prop_forward_ry(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);

void prop_forward_rx(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);

void prop_forward_ry_rx(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);


void prop_backward(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);

void prop_backward_ry(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);

void prop_backward_rx(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);

void prop_backward_ry_rx(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x);

#endif
