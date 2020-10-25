#include "stdint.h"
#include <cmath>

#include <chrono>

#include <limits>
using std::numeric_limits;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

#include <iomanip>
using std::fixed;
using std::setprecision;
using std::setw;

#include <random>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/random.hxx"

#define REPEATS 16

#ifdef POOL_TEST
template <class T>
void print_array(string name, T *array, int32_t batch_size, int32_t size_y, int32_t size_x) {
    int32_t x_counter = 0;
    int32_t batch_counter = 0;

    cout << name << ":" << endl;
    for (int32_t i = 0; i < batch_size * size_y * size_x; i++) {
        cout << fixed << setw(10) << setprecision(3) << array[i];

        x_counter++;
        batch_counter++;

        if (x_counter == size_x) {
            cout << endl;
            x_counter = 0;
        }

        if (batch_counter == (size_x * size_y)) {
            cout << endl;
            batch_counter = 0;
        }
    }
    cout << endl << endl;
}

template <class T>
void print_vector(string name, vector<T> v) {
    cout << name << ":";

    for (int32_t i = 0; i < v.size(); i++) {
        cout << " " << v[i];
    }
    cout << endl;
}
#endif


/********************************************
 * POOL INITIALIZATION
 ********************************************/

void update_offset(vector<int> &pools, vector<int> &offset) {
    offset.clear();

    offset.push_back(0);
    for (int32_t i = 0; i < pools.size() - 1; i++) {
        offset.push_back(offset[i] + pools[i]);
    }
}

void initialize_pools(vector<int> &pools, vector<int> &offset, int input_size, int output_size) {
    pools.clear();

    int min_value;
    int larger_count;
    if (input_size >= output_size) {
        min_value = input_size / output_size;
        larger_count = input_size % output_size;

        pools.insert(pools.end(), larger_count, min_value + 1);
        pools.insert(pools.end(), output_size - larger_count, min_value);
    } else {
        min_value = output_size / input_size;
        larger_count = output_size % input_size;

        pools.insert(pools.end(), larger_count, min_value + 1);
        pools.insert(pools.end(), input_size - larger_count, min_value);
    }

    update_offset(pools, offset);
}


/********************************************
 * FORWARD PROPAGATION
 ********************************************/

//pool forward when the y dimension of the output is less than the y dimension of the input and
//the x dimension of the output is less than the the x dimension of the input
void pool_forward(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t out_y = 0; out_y < y_pools.size(); out_y++) {
            for (int32_t out_x = 0; out_x < x_pools.size(); out_x++) {
                int32_t max_y = 0;
                int32_t max_x = 0;

                int32_t in_y = y_pool_offset[out_y];
                int32_t in_x = x_pool_offset[out_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float max_value = -numeric_limits<float>::max();
                for (int32_t pool_y = 0; pool_y < y_pools[out_y]; pool_y++) {
                    for (int32_t pool_x = 0; pool_x < x_pools[out_x]; pool_x++) {
                        float current_value = input[input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x + pool_x];
                        //cout << " " << current_value;
                        if (current_value > max_value) {
                            max_value = current_value;
                            max_y = pool_y;
                            max_x = pool_x;
                        }
                    }
                }
                //cout << " -- max value: " << max_value << endl;

                for (int32_t pool_y = 0; pool_y < y_pools[out_y]; pool_y++) {
                    for (int32_t pool_x = 0; pool_x < x_pools[out_x]; pool_x++) {
                        if (pool_y == max_y && pool_x == max_x) {
                            output[output_batch_offset + (out_y * output_size_x) + out_x] += max_value * scale;

                            pool_gradients[input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x + pool_x] = scale;
                        } else {
                            pool_gradients[input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x + pool_x] = 0;
                        }
                    }
                }
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
}


void pool_forward(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling) {
    if (training || max_pooling) {
        pool_forward(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
    } else {
        int output_image_size = output_size_y * output_size_x;
        float *temp_output = new float[batch_size * output_image_size];
        std::fill_n(temp_output, batch_size * output_image_size, 0);

        for (int32_t i = 0; i < REPEATS; i++) {
            fisher_yates_shuffle(generator, y_pools);
            fisher_yates_shuffle(generator, x_pools);
            update_offset(y_pools, y_pool_offset);
            update_offset(x_pools, x_pool_offset);

            pool_forward(input, scale, pool_gradients, temp_output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        }

        for (int32_t i = 0; i < batch_size * output_image_size; i++) {
            output[i] += temp_output[i] / REPEATS;
        }

        delete [] temp_output;
    }
}


//pool forward when the y dimension of the output is greater than the y dimension of the input and
//the x dimension of the output is less than the the x dimension of the input
void pool_forward_ry(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t in_y = 0; in_y < input_size_y; in_y++) {
            for (int32_t out_x = 0; out_x < x_pools.size(); out_x++) {
                int32_t max_x = 0;

                int32_t out_y = y_pool_offset[in_y];
                int32_t in_x = x_pool_offset[out_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float max_value = -numeric_limits<float>::max();
                for (int32_t pool_x = 0; pool_x < x_pools[out_x]; pool_x++) {
                    float current_value = input[input_batch_offset + (in_y  * input_size_x) + in_x + pool_x];
                    //cout << " " << current_value;
                    if (current_value > max_value) {
                        max_value = current_value;
                        max_x = pool_x;
                    }
                }
                //cout << " -- max value: " << max_value << endl;

                for (int32_t pool_x = 0; pool_x < x_pools[out_x]; pool_x++) {
                    if (pool_x == max_x) {
                        for (int32_t pool_y = 0; pool_y < y_pools[in_y]; pool_y++) {
                            //cout << "setting output[" << out_y + pool_y << "][" << out_x << "] to: " << max_value << endl;
                            output[output_batch_offset + ((out_y + pool_y) * output_size_x) + out_x] += max_value * scale;
                        }

                        //cout << "setting pool_gradient[" << in_y << "][" << in_x + pool_x << "]: to 1" << endl;
                        pool_gradients[input_batch_offset + (in_y * input_size_x) + in_x + pool_x] = scale;
                    } else {
                        //cout << "setting pool_gradient[" << in_y << "][" << in_x + pool_x << "]: to 0" << endl;
                        pool_gradients[input_batch_offset + (in_y * input_size_x) + in_x + pool_x] = 0;
                    }
                }
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
}

void pool_forward_ry(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling) {
    if (training || max_pooling) {
        pool_forward_ry(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
    } else {
        int output_image_size = output_size_y * output_size_x;
        float *temp_output = new float[batch_size * output_image_size];
        std::fill_n(temp_output, batch_size * output_image_size, 0);

        for (int32_t i = 0; i < REPEATS; i++) {
            fisher_yates_shuffle(generator, y_pools);
            fisher_yates_shuffle(generator, x_pools);
            update_offset(y_pools, y_pool_offset);
            update_offset(x_pools, x_pool_offset);

            pool_forward_ry(input, scale, pool_gradients, temp_output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        }

        for (int32_t i = 0; i < batch_size * output_image_size; i++) {
            output[i] += temp_output[i] / REPEATS;
        }

        delete [] temp_output;
    }
}



//pool forward when the y dimension of the output is less than the y dimension of the input and
//the x dimension of the output is greater than the the x dimension of the input
void pool_forward_rx(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t out_y = 0; out_y < y_pools.size(); out_y++) {
            for (int32_t in_x = 0; in_x < input_size_x; in_x++) {
                int32_t max_y = 0;

                int32_t in_y = y_pool_offset[out_y];
                int32_t out_x = x_pool_offset[in_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float max_value = -numeric_limits<float>::max();
                for (int32_t pool_y = 0; pool_y < y_pools[out_y]; pool_y++) {
                    float current_value = input[input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x];
                    //cout << " " << current_value;
                    if (current_value > max_value) {
                        max_value = current_value;
                        max_y = pool_y;
                    }
                }
                //cout << " -- max value: " << max_value << endl;

                for (int32_t pool_y = 0; pool_y < y_pools[out_y]; pool_y++) {
                    if (pool_y == max_y) {
                        for (int32_t pool_x = 0; pool_x < x_pools[in_x]; pool_x++) {
                            //cout << "setting output[" << out_y << "][" << out_x + pool_x << "] to: " << max_value << endl;
                            output[output_batch_offset + (out_y * output_size_x) + out_x + pool_x] += max_value * scale;
                        }

                        //cout << "setting pool_gradient[" << in_y + pool_y << "][" << in_x << "]: to 1" << endl;
                        pool_gradients[input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x] = scale;
                    } else {
                        //cout << "setting pool_gradient[" << in_y + pool_y << "][" << in_x << "]: to 0" << endl;
                        pool_gradients[input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x] = 0;
                    }
                }
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
}

void pool_forward_rx(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling) {
    if (training || max_pooling) {
        pool_forward_rx(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
    } else {
        int output_image_size = output_size_y * output_size_x;
        float *temp_output = new float[batch_size * output_image_size];
        std::fill_n(temp_output, batch_size * output_image_size, 0);

        for (int32_t i = 0; i < REPEATS; i++) {
            fisher_yates_shuffle(generator, y_pools);
            fisher_yates_shuffle(generator, x_pools);
            update_offset(y_pools, y_pool_offset);
            update_offset(x_pools, x_pool_offset);

            pool_forward_rx(input, scale, pool_gradients, temp_output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        }

        for (int32_t i = 0; i < batch_size * output_image_size; i++) {
            output[i] += temp_output[i] / REPEATS;
        }

        delete [] temp_output;
    }
}




//pool forward when the y dimension of the output is greater than the y dimension of the input and
//the x dimension of the output is greater than the the x dimension of the input
void pool_forward_ry_rx(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t in_y = 0; in_y < input_size_y; in_y++) {
            for (int32_t in_x = 0; in_x < input_size_x; in_x++) {
                int32_t out_y = y_pool_offset[in_y];
                int32_t out_x = x_pool_offset[in_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float max_value = input[input_batch_offset + (in_y * input_size_x) + in_x];
                //cout << " -- max value: " << max_value << endl;

                for (int32_t pool_y = 0; pool_y < y_pools[in_y]; pool_y++) {
                    for (int32_t pool_x = 0; pool_x < x_pools[in_x]; pool_x++) {
                        //cout << "setting output[" << out_y + pool_y << "][" << out_x + pool_x << "] to: " << max_value << endl;
                        output[output_batch_offset + ((out_y + pool_y) * output_size_x) + out_x + pool_x] += max_value * scale;
                    }
                }
                //cout << "setting pool_gradient[" << in_y << "][" << in_x << "]: to 1" << endl;
                pool_gradients[input_batch_offset + (in_y * input_size_x) + in_x] = scale;
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
}

void pool_forward_ry_rx(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling) {
    if (training || max_pooling) {
        pool_forward_ry_rx(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
    } else {
        int output_image_size = output_size_y * output_size_x;
        float *temp_output = new float[batch_size * output_image_size];
        std::fill_n(temp_output, batch_size * output_image_size, 0);

        for (int32_t i = 0; i < REPEATS; i++) {
            fisher_yates_shuffle(generator, y_pools);
            fisher_yates_shuffle(generator, x_pools);
            update_offset(y_pools, y_pool_offset);
            update_offset(x_pools, x_pool_offset);

            pool_forward_ry_rx(input, scale, pool_gradients, temp_output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);
        }

        for (int32_t i = 0; i < batch_size * output_image_size; i++) {
            output[i] += temp_output[i] / REPEATS;
        }

        delete [] temp_output;
    }
}



/********************************************
 * BACK PROPAGATION
 ********************************************/


//pool backward when the y dimension of the output is less than the y dimension of the input and
//the x dimension of the output is less than the the x dimension of the input
void pool_backward(float* input_errors, float &scale_update, const float *inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    scale_update = 0.0;
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t out_y = 0; out_y < y_pools.size(); out_y++) {
            for (int32_t out_x = 0; out_x < x_pools.size(); out_x++) {
                int32_t in_y = y_pool_offset[out_y];
                int32_t in_x = x_pool_offset[out_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float output_error = output_errors[output_batch_offset + (out_y * output_size_x) + out_x];

                for (int32_t pool_y = 0; pool_y < y_pools[out_y]; pool_y++) {
                    for (int32_t pool_x = 0; pool_x < x_pools[out_x]; pool_x++) {
                        int position = input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x + pool_x;
                        float delta = output_error * pool_gradients[position]; //pool gradients includes scale
                        input_errors[position] += delta;
                        scale_update += inputs[position] * delta;
                    }
                }
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
    scale_update /= batch_size;
}

//pool backward when the y dimension of the output is greater than the y dimension of the input and
//the x dimension of the output is less than the the x dimension of the input
void pool_backward_ry(float* input_errors, float &scale_update, const float *inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    scale_update = 0.0;
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t in_y = 0; in_y < input_size_y; in_y++) {
            for (int32_t out_x = 0; out_x < x_pools.size(); out_x++) {
                int32_t out_y = y_pool_offset[in_y];
                int32_t in_x = x_pool_offset[out_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float output_error = 0.0;
                for (int32_t pool_y = 0; pool_y < y_pools[in_y]; pool_y++) {
                    output_error += output_errors[output_batch_offset + ((out_y + pool_y) * output_size_x) + out_x];
                }

                for (int32_t pool_x = 0; pool_x < x_pools[out_x]; pool_x++) {
                    int position = input_batch_offset + (in_y * input_size_x) + in_x + pool_x;
                    float delta = output_error * pool_gradients[position];
                    input_errors[position] += delta;
                    scale_update += inputs[position] * delta;
                }
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
    scale_update /= batch_size;
}

//pool backward when the y dimension of the output is less than the y dimension of the input and
//the x dimension of the output is greater than the the x dimension of the input
void pool_backward_rx(float* input_errors, float &scale_update, const float *inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    scale_update = 0.0;
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t out_y = 0; out_y < y_pools.size(); out_y++) {
            for (int32_t in_x = 0; in_x < input_size_x; in_x++) {
                int32_t in_y = y_pool_offset[out_y];
                int32_t out_x = x_pool_offset[in_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float output_error = 0.0;
                for (int32_t pool_x = 0; pool_x < x_pools[in_x]; pool_x++) {
                    output_error += output_errors[output_batch_offset + (out_y * output_size_x) + out_x + pool_x];
                }

                for (int32_t pool_y = 0; pool_y < y_pools[out_y]; pool_y++) {
                    int position = input_batch_offset + ((in_y + pool_y) * input_size_x) + in_x;
                    float delta = output_error * pool_gradients[position];
                    input_errors[position] += delta;
                    scale_update += inputs[position] * delta;
                }
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
    scale_update /= batch_size;
}

//pool backward when the y dimension of the output is greater than the y dimension of the input and
//the x dimension of the output is greater than the the x dimension of the input
void pool_backward_ry_rx(float* input_errors, float &scale_update, const float *inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset) {
    int32_t input_batch_offset = 0;
    int32_t output_batch_offset = 0;

    int32_t input_image_size = input_size_y * input_size_x;
    int32_t output_image_size = output_size_y * output_size_x;

    scale_update = 0.0;
    for (int32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        for (int32_t in_y = 0; in_y < input_size_y; in_y++) {
            for (int32_t in_x = 0; in_x < input_size_x; in_x++) {
                int32_t out_y = y_pool_offset[in_y];
                int32_t out_x = x_pool_offset[in_x];

                //cout << "getting max value for pool starting at " << in_y  << " x " << in_x << " from:" << endl;
                float output_error = 0.0;
                for (int32_t pool_x = 0; pool_x < x_pools[in_x]; pool_x++) {
                    for (int32_t pool_y = 0; pool_y < y_pools[in_y]; pool_y++) {
                        output_error += output_errors[output_batch_offset + ((out_y + pool_y) * output_size_x) + out_x + pool_x];
                    }
                }

                //cout << "setting input error[" << in_y << "][" << in_x << "] to: " << output_error << endl;

                int position = input_batch_offset + (in_y * input_size_x) + in_x;
                float delta = output_error * pool_gradients[position];
                input_errors[position] += delta;
                scale_update += inputs[position] * delta;
            }
        }
        input_batch_offset += input_image_size;
        output_batch_offset += output_image_size;
    }
    scale_update /= batch_size;
}


#ifdef POOL_TEST

void test_pooling(int input_size_y, int input_size_x, int output_size_y, int output_size_x, int batch_size, minstd_rand0 &generator, bool training) {
    float *input = new float[batch_size * input_size_y * input_size_x];
    float *input_errors = new float[batch_size * input_size_y * input_size_x];
    float *pool_gradients = new float[batch_size * input_size_y * input_size_x];
    for (int32_t i = 0; i < batch_size * input_size_y * input_size_x; i++) {
        input[i] = ceil(100 * drand48());
        input_errors[i] = 0;
        pool_gradients[i] = 0;
    }

    float *output = new float[batch_size * output_size_y * output_size_x];
    float *output_errors = new float[batch_size * output_size_y * output_size_x];
    for (int32_t i = 0; i < batch_size * output_size_y * output_size_x; i++) {
        output[i] = 0.0;
        output_errors[i] = ceil(100 * drand48());
    }

    print_array("input", input, batch_size, input_size_y, input_size_x);
    print_array("output", output, batch_size, output_size_y, output_size_x);
    print_array("pool_gradients", pool_gradients, batch_size, input_size_y, input_size_x);
    print_array("input_errors", input_errors, batch_size, input_size_y, input_size_x);
    print_array("output_errors", output_errors, batch_size, output_size_y, output_size_x);


    vector<int> y_pools;
    vector<int> y_pool_offset;
    initialize_pools(y_pools, y_pool_offset, input_size_y, output_size_y);

    vector<int> x_pools;
    vector<int> x_pool_offset;
    initialize_pools(x_pools, x_pool_offset, input_size_x, output_size_x);

    print_vector("y_pools (before shuffle)", y_pools);
    print_vector("x_pools (before shuffle)", x_pools);

    print_vector("y_pool_offset (before shuffle)", y_pool_offset);
    print_vector("x_pool_offset (before shuffle)", x_pool_offset);

    fisher_yates_shuffle(generator, y_pools);
    fisher_yates_shuffle(generator, x_pools);
    update_offset(y_pools, y_pool_offset);
    update_offset(x_pools, x_pool_offset);

    print_vector("y_pools (after shuffle)", y_pools);
    print_vector("x_pools (after shuffle)", x_pools);

    print_vector("y_pool_offset (after shuffle)", y_pool_offset);
    print_vector("x_pool_offset (after shuffle)", x_pool_offset);

    float scale = (10.0 * drand48()) - 5.0;
    float scale_update;

    if (input_size_y >= output_size_y && input_size_x >= output_size_x) {
        pool_forward(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, false);
        pool_backward(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);

    } else if (input_size_x >= output_size_x) {
        pool_forward_ry(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, false);
        pool_backward_ry(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);

    } else if (input_size_y >= output_size_y) {
        pool_forward_rx(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, false);
        pool_backward_rx(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);

    } else {
        pool_forward_ry_rx(input, scale, pool_gradients, output, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset, generator, training, false);
        pool_backward_ry_rx(input_errors, scale_update, input, pool_gradients, output_errors, batch_size, input_size_y, input_size_x, output_size_y, output_size_x, y_pools, x_pools, y_pool_offset, x_pool_offset);

    }
    cout << "scale_update: " << scale_update << endl;

    print_array("output", output, batch_size, output_size_y, output_size_x);

    print_array("pool_gradients", pool_gradients, batch_size, input_size_y, input_size_x);
    print_array("input_errors", input_errors, batch_size, input_size_y, input_size_x);
    print_array("output_errors", output_errors, batch_size, output_size_y, output_size_x);

    delete [] input;
    delete [] output;
    delete [] pool_gradients;
    delete [] input_errors;
    delete [] output_errors;
}

int main(int argc, char **argv) {
    int batch_size = 3;

    cout << "TESTING REGULAR POOLING:" << endl;

    bool training = false;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);

    // regular
    int input_size_y = 13;
    int input_size_x = 15;
    int output_size_y = 6;
    int output_size_x = 4;
    test_pooling(input_size_y, input_size_x, output_size_y, output_size_x, batch_size, generator, training);

    cout << endl << endl << endl;
    cout << "TESTING POOLING WITH REVERSE FILTER Y:" << endl;
    //reverse filter y
    input_size_y = 6;
    input_size_x = 15;
    output_size_y = 13;
    output_size_x = 4;
    test_pooling(input_size_y, input_size_x, output_size_y, output_size_x, batch_size, generator, training);

    cout << endl << endl << endl;
    cout << "TESTING POOLING WITH REVERSE FILTER X:" << endl;
    //reverse filter x
    input_size_y = 13;
    input_size_x = 4;
    output_size_y = 6;
    output_size_x = 15;
    test_pooling(input_size_y, input_size_x, output_size_y, output_size_x, batch_size, generator, training);

    cout << endl << endl << endl;
    cout << "TESTING POOLING WITH REVERSE FILTER Y and X:" << endl;
    //reverse filter x and reverse filter y
    input_size_y = 6;
    input_size_x = 4;
    output_size_y = 13;
    output_size_x = 15;
    test_pooling(input_size_y, input_size_x, output_size_y, output_size_x, batch_size, generator, training);
}


#endif
