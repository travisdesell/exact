#pragma unroll

#define input_y 28
#define input_x 28
#define weight_y 15
#define weight_x 15
#define output_y 14
#define output_x 14
#define input_image_size 784
#define output_image_size 196

__kernel void propagate_forward(const __global float *inputs, __constant float *weights, __global float *outputs, __local float *inputs_local) {
    const int batch_number = get_group_id(0);
    int y = get_global_id(0);
    const int x = get_global_id(1);

    const int output_position = (y * output_x) + x;

    //printf("thread[%d][%d][%d] output position: %d\n", batch_number, y, x, output_position);
    y = y % output_y;

    const int inputs_local_offset = ((y * output_x) + x) * 4;
    const int input_offset = (batch_number * input_image_size) + inputs_local_offset;
    inputs_local[inputs_local_offset]       = inputs[input_offset ];
    inputs_local[inputs_local_offset + 1]   = inputs[input_offset + 1];
    inputs_local[inputs_local_offset + 2]   = inputs[input_offset + 2];
    inputs_local[inputs_local_offset + 3]   = inputs[input_offset + 3];
    barrier(CLK_LOCAL_MEM_FENCE);

    float output = outputs[output_position];
    //float output = 0.0;

    int current_input = (y * input_x) + x;
    int current_weight = 0;
    const int width_difference = (input_x - output_x) - 1;

    for (int wy = 0; wy < weight_y; wy++) {
        for (int wx = 0; wx < weight_x; wx++) {
            output += weights[current_weight++] * inputs_local[current_input++];
        }
        current_input += width_difference;
    }

    outputs[output_position] = output;
}
