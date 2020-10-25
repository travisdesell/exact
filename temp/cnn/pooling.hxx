#ifndef EXACT_POOLING_HXX
#define EXACT_POOLING_HXX

#include <vector>
using std::vector;

void update_offset(vector<int> &pools, vector<int> &offset);
void initialize_pools(vector<int> &pools, vector<int> &offset, int input_size, int output_size);


void pool_forward(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling);

void pool_forward_ry(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling);

void pool_forward_rx(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling);

void pool_forward_ry_rx(const float* input, float scale, float *pool_gradients, float* output, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset, minstd_rand0 &generator, bool training, bool max_pooling);


void pool_backward(float* input_errors, float &scale_update, const float* inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset);

void pool_backward_ry(float* input_errors, float &scale_update, const float* inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset);

void pool_backward_rx(float* input_errors, float &scale_update, const float* inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset);

void pool_backward_ry_rx(float* input_errors, float &scale_update, const float* inputs, const float *pool_gradients, const float* output_errors, int32_t batch_size, int32_t input_size_y, int32_t input_size_x, int32_t output_size_y, int32_t output_size_x, vector<int> &y_pools, vector<int> &x_pools, vector<int> &y_pool_offset, vector<int> &x_pool_offset);

#endif
