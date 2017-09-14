#include "stdint.h"
#include <cmath>

#include <iostream>
using std::cerr;
using std::endl;

#include <vector>
using std::vector;

#include "propagation.hxx"

void prop_forward(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;
    int width_difference = input_size_x - output_size_x;

#ifdef NAN_CHECKS
    double previous_output;
#endif

    current_output = 0;
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                float weight = weights[current_weight++];

                current_output = (batch_number * output_image_size);
                current_input = (batch_number * input_image_size) + (fy * input_size_x) + fx;

                for (int32_t y = 0; y < output_size_y; y++) {
                    for (int32_t x = 0; x < output_size_x; x++) {
#ifdef NAN_CHECKS
                        previous_output = output[current_output];
#endif
                        output[current_output++] += weight * input[current_input++];

#ifdef NAN_CHECKS
                        if (isnan(output[current_output - 1]) || isinf(output[current_output - 1])) {
                            cerr << "ERROR! NAN or INF in propagate forward" << endl;
                            cerr << "previous_output: " << previous_output << ", output: " << output[current_output - 1] << endl;
                            cerr << "weight: " << weight << ", input: " << input[current_input - 1] << endl;
                            exit(1);
                        }
#endif
                    }
                    current_input += width_difference;
                }
            }
        }
    }
}


void prop_forward_ry(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;
    //int width_difference = input_size_x - output_size_x;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                float weight = weights[current_weight++];

                for (int32_t y = 0; y < input_size_y; y++) {
                    current_input = (batch_number * input_image_size) + (input_size_x * y) + fx;
                    current_output = (batch_number * output_image_size) + (output_size_x * (fy + y));
                    for (int32_t x = 0; x < output_size_x; x++) {
                        float value = weight * input[current_input + x];
                        output[current_output + x] += value;
                    }
                }
            }
        }
    }
}

void prop_forward_rx(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;
    //int width_difference = input_size_x - output_size_x;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                float weight = weights[current_weight++];

                for (int32_t y = 0; y < output_size_y; y++) {
                    current_input = (batch_number * input_image_size) + (input_size_x * (fy + y));
                    current_output = (batch_number * output_image_size) + (output_size_x * y) + fx;

                    for (int32_t x = 0; x < input_size_x; x++) {
                        float value = weight * input[current_input + x];
                        output[current_output + x] += value;
                    }
                }
            }
        }
    }
}


void prop_forward_ry_rx(const float* input, const float* weights, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;
    //int width_difference = input_size_x - output_size_x;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                float weight = weights[current_weight++];

                for (int32_t y = 0; y < input_size_y; y++) {
                    current_input = (batch_number * input_image_size) + (input_size_x * y);
                    current_output = (batch_number * output_image_size) + (output_size_x * (fy + y)) + fx;
                    for (int32_t x = 0; x < input_size_x; x++) {
                        float value = weight * input[current_input + x];
                        output[current_output + x] += value;
                    }
                }
            }
        }
    }
}


void prop_backward(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;
    int width_difference = input_size_x - output_size_x;

    float weight_update, weight, delta;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[current_weight];

                current_output = (batch_number * output_image_size);
                current_input = (batch_number * input_image_size) + (fy * input_size_x) + fx;

                for (int32_t y = 0; y < output_size_y; y++) {
                    for (int32_t x = 0; x < output_size_x; x++) {
                        delta = output_errors[current_output++];

                        weight_update += input[current_input] * delta;
                        input_errors[current_input] += delta * weight;
                        current_input++;
                    }
                    current_input += width_difference;
                }
                weight_updates[current_weight] += weight_update / batch_size;
                current_weight++;
            }
        }
    }
}

void prop_backward_ry(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;

    float weight_update, weight, delta;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[current_weight];

                for (int32_t y = 0; y < input_size_y; y++) {
                    current_output = (batch_number * output_image_size) + (output_size_x * (fy + y));
                    current_input = (batch_number * input_image_size) + (input_size_x * y) + fx;

                    for (int32_t x = 0; x < output_size_x; x++) {
                        delta = output_errors[current_output + x];
                        weight_update += input[current_input + x] * delta;
                        input_errors[current_input + x] += delta * weight;
                    }
                }
                weight_updates[current_weight] += weight_update / batch_size;
                current_weight++;
            }
        }
    }
}

void prop_backward_rx(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;

    float weight_update, weight, delta;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0;
                weight = weights[current_weight];

                for (int32_t y = 0; y < output_size_y; y++) {
                    current_output = (batch_number * output_image_size) + (output_size_x * y) + fx;
                    current_input = (batch_number * input_image_size) + (input_size_x * (fy + y));

                    for (int32_t x = 0; x < input_size_x; x++) {
                        delta = output_errors[current_output + x];
                        weight_update += input[current_input + x] * delta;
                        input_errors[current_input + x] += delta * weight;
                    }
                }
                weight_updates[current_weight] += weight_update / batch_size;
                current_weight++;
            }
        }
    }
}

void prop_backward_ry_rx(float* output_errors, float* input, float* input_errors, float* weight_updates, float* weights, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t filter_y, int32_t filter_x, int32_t output_size_y, int32_t output_size_x) {
    int current_weight, current_output, current_input;

    int output_image_size = output_size_y * output_size_x;
    int input_image_size = input_size_y * input_size_x;

    float weight_update, weight, delta;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (int32_t fy = 0; fy < filter_y; fy++) {
            for (int32_t fx = 0; fx < filter_x; fx++) {
                weight_update = 0.0;
                weight = weights[current_weight];

                for (int32_t y = 0; y < input_size_y; y++) {
                    current_output = (batch_number * output_image_size) + (output_size_x * (fy + y)) + fx;
                    current_input = (batch_number * input_image_size) + (input_size_x * y);

                    for (int32_t x = 0; x < input_size_x; x++) {
                        delta = output_errors[current_output + x];
                        weight_update += input[current_input + x] * delta;
                        input_errors[current_input + x] += delta * weight;
                    }
                }
                weight_updates[current_weight] += weight_update / batch_size;
                current_weight++;
            }
        }
    }
}

#ifdef PROPAGATE_TEST
int main(int argc, char **argv) {
    //make tests for the 8 convolve operations

}
#endif
