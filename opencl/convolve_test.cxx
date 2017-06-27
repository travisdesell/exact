#include <cmath>
using std::sqrt;

#include <chrono>

#include <string>
using std::string;

#include <vector>
using std::vector;

#include "common/arguments.hxx"

#include "image_tools/image_set.hxx"

#ifdef __OPENCL__

#include <cstdio>
#include <cstring>
#include <time.h>

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "opencl_utils.hxx"

#endif

cl_device_id device;
cl_context context;
cl_program propagate_forward_kernel;
cl_kernel kernel;
cl_command_queue queue;

int opencl_input_size;
int opencl_weights_size;
int opencl_output_size;

size_t *global_size;
size_t *local_size;

cl_mem sizes_opencl;
cl_mem input_opencl;
cl_mem weights_opencl;
cl_mem output_opencl;


#define PROPAGATE_FORWARD_KERNEL_FILE "../opencl/propagate_forward_kernel.cl"

string generate_program(int batch_size, int input_y, int input_x, int weights_y, int weights_x, int output_y, int output_x) {
    string program;

    program.append("#pragma unroll\n");

    program.append("#define input_y " + to_string(input_y) + "\n");
    program.append("#define input_x " + to_string(input_x) + "\n");
    program.append("#define weight_y " + to_string(weights_y) + "\n");
    program.append("#define weight_x " + to_string(weights_x) + "\n");
    program.append("#define output_y " + to_string(output_y) + "\n");
    program.append("#define output_x " + to_string(output_x) + "\n");
    program.append("#define input_image_size " + to_string(input_y * input_x) + "\n");
    program.append("#define output_image_size " + to_string(output_y * output_x) + "\n");

    program.append("__kernel void propagate_forward(const __global float *inputs, __constant float *weights, __global float *outputs, __local float *inputs_local) {\n");
    program.append("    int batch_number = get_group_id(0);\n");
    program.append("    int y = get_global_id(0);\n");
    program.append("    int x = get_global_id(1);\n");

    program.append("    int output_position = (y * output_x) + x;\n");

    program.append("    y = y % output_y;\n");

    int n_local_copies = ceil((float)(input_y * input_x) / (float)(output_y * output_x));

	program.append("    int inputs_local_offset = (y * output_x) + x;\n");
    program.append("    int input_offset = (batch_number * input_image_size) + inputs_local_offset;\n");

    program.append("    inputs_local[inputs_local_offset]       = inputs[input_offset];\n");
    //program.append("    printf(\"thread[%d,%d,%d] inputs_local[%d] = inputs[%d]: %f\\n\", batch_number, y, x, inputs_local_offset, input_offset, inputs_local[inputs_local_offset]);\n");
    for (uint32_t i = 1; i < n_local_copies; i++) {
        if (i == n_local_copies - 1 && (input_y * input_x) % (output_y * output_x) != 0) {
            program.append("    if (inputs_local_offset + " + to_string(i * output_y * output_x) + " < input_image_size) {\n");
            program.append("        inputs_local[inputs_local_offset + " + to_string(i * output_y * output_x) + "]   = inputs[input_offset + " + to_string(i * output_y * output_x) + "];\n");
            //program.append("        printf(\"thread[%d,%d,%d] inputs_local[%d] = inputs[%d]: %f -- IN IF STATEMENT\\n\", batch_number, y, x, inputs_local_offset + " + to_string(i * output_y * output_x) + ", input_offset + " + to_string(i * output_y * output_x) + ", inputs_local[inputs_local_offset + " + to_string(i * output_y * output_x) + "]);\n");
            program.append("    }\n");
        } else {
            program.append("    inputs_local[inputs_local_offset + " + to_string(i * output_y * output_x) + "]   = inputs[input_offset + " + to_string(i * output_y * output_x) + "];\n");
            //program.append("    printf(\"thread[%d,%d,%d] inputs_local[%d] = inputs[%d]: %f\\n\", batch_number, y, x, inputs_local_offset + " + to_string(i * output_y * output_x) + ", input_offset + " + to_string(i * output_y * output_x) + ", inputs_local[inputs_local_offset + " + to_string(i * output_y * output_x) + "]);\n");
        }
    }

	program.append("    barrier(CLK_LOCAL_MEM_FENCE);\n");

	//program.append("    float output = outputs[output_position];\n");
    program.append("    float output = 0.0;\n");
    program.append("    int local_input_offset = (y * input_x) + x;\n");

    int current_weight = 0;
    int current_input = 0;
    for (uint32_t wy = 0; wy < weights_y; wy++) {
        for (uint32_t wx = 0; wx < weights_x; wx++) {
            program.append("    output += weights[" + to_string(current_weight++) + "] * inputs_local[local_input_offset + " + to_string(current_input++) + "];\n");
        }
        current_input += input_x - weights_x;
    }

    program.append("    outputs[output_position] = output;\n");
    //program.append("    printf(\"outputs[%d] = %f\\n\", output_position, output);\n");
    program.append("}\n");

    return program;
}


void initialize_opencl(int batch_size, int input_y, int input_x, int weights_y, int weights_x, int output_y, int output_x) {
    //OpenCL structures
    cl_int err;

    //Create device and context
    device = create_device();
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    check_error(err, "couldn't create a context, err: %d", err);

    //Create a command queue
    queue = clCreateCommandQueue(context, device, 0, &err);
    check_error(err, "couldn't create a command queue: %d", err);

    // Build program
    //propagate_forward_kernel = build_program(context, device, PROPAGATE_FORWARD_KERNEL_FILE);

    string program = generate_program(batch_size, input_y, input_x, weights_y, weights_x, output_y, output_x);

	cout << "KERNEL OUTSIDE:" << endl;
    cout << program;
    cout << endl << endl;

    const char* program_c_str = program.c_str();
    propagate_forward_kernel = clCreateProgramWithSource(context, 1, (const char**)&program_c_str, NULL, &err);
    check_error(err, "could not create program: %d", err);
    if (!propagate_forward_kernel) {
        cerr << "ERROR couldn't craete propagate_forward kernel!" << endl;
        exit(1);
    }


    err = clBuildProgram(propagate_forward_kernel, 1, &device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t len;
        char buffer[2048];
        printf("Error: Failed to build program executable!\n");
        clGetProgramBuildInfo(propagate_forward_kernel, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
        printf("%s\n", buffer);
    }
    check_error(err, "could not build program: %d", err);

    opencl_input_size = sizeof(float) * batch_size * input_y * input_x;
    input_opencl = clCreateBuffer(context, CL_MEM_READ_ONLY, opencl_input_size, NULL, &err);
    check_error(err, "could not create input_opencl buffer: %d", err);

    opencl_weights_size = sizeof(float) * weights_y * weights_x;
    weights_opencl = clCreateBuffer(context, CL_MEM_READ_ONLY, opencl_weights_size, NULL, &err);
    check_error(err, "could not create weights_opencl buffer: %d", err);

    opencl_output_size = sizeof(float) * batch_size * output_y * output_x;
    output_opencl = clCreateBuffer(context, CL_MEM_READ_WRITE, opencl_output_size, NULL, &err);
    check_error(err, "could not create output_opencl buffer: %d", err);

    // Create a kernel
    kernel = clCreateKernel(propagate_forward_kernel, "propagate_forward", &err);
    check_error(err, "couldn't create a kernel: %d", err);

    // Create kernel arguments
    err =  clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_opencl);
    check_error(err, "couldn't create input_opencl argument: %d", err);

    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &weights_opencl);
    check_error(err, "couldn't create weights_opencl argument: %d", err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &output_opencl);
    check_error(err, "couldn't create output_opencl argument: %d", err);

    err = clSetKernelArg(kernel, 3, sizeof(cl_float) * input_y * input_x, NULL);

    global_size = (size_t*)malloc(sizeof(size_t) * 2);
    global_size[0] = batch_size * output_y;
    global_size[1] = output_x;

    local_size = (size_t*)malloc(sizeof(size_t) * 2);
    local_size[0] = output_y;
    local_size[1] = output_x;
}


void initialize_3d(float ****v, uint32_t v_z, uint32_t v_y, uint32_t v_x) {
    (*v) = (float***)malloc(sizeof(float**) * v_z);
    for (uint32_t z = 0; z < v_z; z++) {
        (*v)[z] = (float**)malloc(sizeof(float*) * v_y);
        for (uint32_t y = 0; y < v_y; y++) {
            (*v)[z][y] = (float*)malloc(sizeof(float) * v_x);
        }
    }
}

void initialize_2d(float ***v, uint32_t v_y, uint32_t v_x) {
    *v = (float**)malloc(sizeof(float*) * v_y);
    for (uint32_t y = 0; y < v_y; y++) {
        (*v)[y] = (float*)malloc(sizeof(float) * v_x);
    }
}


void set_to_random_3d(float ***v, uint32_t v_z, uint32_t v_y, uint32_t v_x) {
    for (uint32_t z = 0; z < v_z; z++) {
        for (uint32_t y = 0; y < v_y; y++) {
            for (uint32_t x = 0; x < v_x; x++) {
                v[z][y][x] = drand48();
            }
        }
    }
}

void set_to_random_2d(float **v, uint32_t v_y, uint32_t v_x) {
    for (uint32_t y = 0; y < v_y; y++) {
        for (uint32_t x = 0; x < v_x; x++) {
            v[y][x] = drand48();
        }
    }
}


void set_to_zero_3d(float ***v, uint32_t v_z, uint32_t v_y, uint32_t v_x) {
    for (uint32_t z = 0; z < v_z; z++) {
        for (uint32_t y = 0; y < v_y; y++) {
            for (uint32_t x = 0; x < v_x; x++) {
                v[z][y][x] = 0.0;
            }
        }
    }
}



void copy_3d_to_1d(float ***input, uint32_t input_z, uint32_t input_y, uint32_t input_x, float *output) {
    uint32_t current_output = 0;
    for (uint32_t z = 0; z < input_z; z++) {
        for (uint32_t y = 0; y < input_y; y++) {
            for (uint32_t x = 0; x < input_x; x++) {
                output[current_output++] = input[z][y][x];
            }
        }
    }
}

void copy_2d_to_1d(float **input, uint32_t input_y, uint32_t input_x, float *output) {
    uint32_t current_output = 0;
    for (uint32_t y = 0; y < input_y; y++) {
        for (uint32_t x = 0; x < input_x; x++) {
            output[current_output++] = input[y][x];
        }
    }
}


void copy_1d_to_3d(float *input, float ***output, uint32_t output_z, uint32_t output_y, uint32_t output_x) {
    uint32_t current_output = 0;
    for (uint32_t z = 0; z < output_z; z++) {
        for (uint32_t y = 0; y < output_y; y++) {
            for (uint32_t x = 0; x < output_x; x++) {
                output[z][y][x] = input[current_output++];
            }
        }
    }
}

bool check_equal(uint32_t batch_size, float ***v1, uint32_t v_y, uint32_t v_x, float *v2) {
    uint32_t current_v2 = 0;
    for (uint32_t z = 0; z < batch_size; z++) {
        for (uint32_t y = 0; y < v_y; y++) {
            for (uint32_t x = 0; x < v_x; x++) {
                if (v1[z][y][x] != v2[current_v2++]) return false;
            }
        }
    }
    return true;
}


void copy_1d_to_2d(float *input, float **output, uint32_t output_y, uint32_t output_x) {
    uint32_t current_output = 0;
    for (uint32_t y = 0; y < output_y; y++) {
        for (uint32_t x = 0; x < output_x; x++) {
            output[y][x] = input[current_output++];
        }
    }
}


void print_3d(float ***input, uint32_t input_z, uint32_t input_y, uint32_t input_x) {
    for (uint32_t z = 0; z < input_z; z++) {
        for (uint32_t y = 0; y < input_y; y++) {
            for (uint32_t x = 0; x < input_x; x++) {
                cout << " " << input[z][y][x];
            }
            cout << endl;
        }
        cout << endl;
    }
}

void print_2d(float **input, uint32_t input_y, uint32_t input_x) {
    for (uint32_t y = 0; y < input_y; y++) {
        for (uint32_t x = 0; x < input_x; x++) {
            cout << " " << input[y][x];
        }
        cout << endl;
    }
}


void print_1d(float *input, uint32_t input_z, uint32_t input_y, uint32_t input_x) {
    uint32_t current_input = 0;

    for (uint32_t z = 0; z < input_z; z++) {
        for (uint32_t y = 0; y < input_y; y++) {
            for (uint32_t x = 0; x < input_x; x++) {
                cout << " " << input[current_input++];
            }
            cout << endl;
        }
        cout << endl;
    }
}

void propagate_forward_3d(uint32_t batch_size, float ***input, uint32_t input_y, uint32_t input_x, float **weights, uint32_t weights_y, uint32_t weights_x, float ***output, uint32_t output_y, uint32_t output_x) {
    float weight;

    for (uint32_t batch_number = 0; batch_number < batch_size; batch_number++) {

        for (uint32_t wy = 0; wy < weights_y; wy++) {
            for (uint32_t wx = 0; wx < weights_x; wx++) {
                weight = weights[wy][wx];

                for (uint32_t y = 0; y < output_y; y++) {
                    for (uint32_t x = 0; x < output_x; x++) {
                        output[batch_number][y][x] += weight * input[batch_number][wy + y][wx + x];
                    }
                }

            }
        }
    }
}

void propagate_forward_1d(uint32_t batch_size, float *input, uint32_t input_y, uint32_t input_x, float *weights, uint32_t weights_y, uint32_t weights_x, float *output, uint32_t output_y, uint32_t output_x) {
    int current_weight = 0;
    int current_output = 0;
    int current_input = 0;
    float weight;

    int input_image_size = input_y * input_x;
    int output_image_size = output_y * output_x;

    int width_difference = input_x - output_x;

    for (uint32_t batch_number = 0; batch_number < batch_size; batch_number++) {
        current_weight = 0;

        for (uint32_t wy = 0; wy < weights_y; wy++) {
            for (uint32_t wx = 0; wx < weights_x; wx++) {
                weight = weights[current_weight++];

                current_output = batch_number * output_image_size;
                current_input = (batch_number * input_image_size) + (wy * input_x) + wx;

                for (uint32_t y = 0; y < output_y; y++) {
                    for (uint32_t x = 0; x < output_x; x++) {
                        output[current_output++] += weight * input[current_input++];
                    }
                    current_input += width_difference;
                }
            }
        }
    }
}


void propagate_forward_opencl(float *input, float *weights, float *output) {
    cl_int err;

    //err = clEnqueueWriteBuffer(queue, input_opencl, CL_TRUE, 0, opencl_input_size, input, 0, NULL, NULL);
    //check_error(err, "couldn't read the output nodes buffer: %d", err);

    //err = clEnqueueWriteBuffer(queue, weights_opencl, CL_TRUE, 0, opencl_weights_size, weights, 0, NULL, NULL);
    //check_error(err, "couldn't read the output nodes buffer: %d", err);

    // Enqueue kernel
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size, 0, NULL, NULL); 
    clFinish(queue);
    //check_error(err, "couldn't enqueue the kernel: %d", err);

    // Read the kernel's output
    //err = clEnqueueReadBuffer(queue, output_opencl, CL_TRUE, 0, opencl_output_size, output, 0, NULL, NULL);
    //check_error(err, "couldn't read the output nodes buffer: %d", err);

    /*
    clEnqueueWriteBuffer(queue, input_opencl, CL_TRUE, 0, opencl_input_size, input, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, weights_opencl, CL_TRUE, 0, opencl_weights_size, weights, 0, NULL, NULL);
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, &global_size, local_size, 0, NULL, NULL); 
    clFinish(queue);
    clEnqueueReadBuffer(queue, output_opencl, CL_TRUE, 0, opencl_output_size, output, 0, NULL, NULL);
    */
}



int main(int argc, char **argv) {
    vector<string> arguments = vector<string>(argv, argv + argc);

    uint32_t batch_size = 1000;

    uint32_t input_y = 10;
    uint32_t input_x = 10;

    uint32_t output_y = 5;
    uint32_t output_x = 5;

    uint32_t weights_y = (input_y - output_y) + 1;
    uint32_t weights_x = (input_x - output_x) + 1;

    float ***input;
    float ***output;
    float **weights;


    float *input_flat = (float*)malloc(sizeof(float) * batch_size * input_y * input_x);
    float *output_flat = (float*)malloc(sizeof(float) * batch_size * output_y * output_x);
    float *weights_flat = (float*)malloc(sizeof(float) * weights_y * weights_x);

    float *input_cpu = (float*)malloc(sizeof(float) * batch_size * input_y * input_x);
    float *output_cpu = (float*)malloc(sizeof(float) * batch_size * output_y * output_x);
    float *weights_cpu = (float*)malloc(sizeof(float) * weights_y * weights_x);

    cout << "input batch size: " << batch_size << ", input_y: " << input_y << ", input_x: " << input_x << endl;
    cout << "output batch size: " << batch_size << ", output_y: " << output_y << ", output_x: " << output_x << endl;
    cout << "weights_y: " << weights_y << ", weights_x: " << weights_x << endl;

    initialize_3d(&input, batch_size, input_y, input_x);
    initialize_3d(&output, batch_size, output_y, output_x);
    initialize_2d(&weights, weights_y, weights_x);

    set_to_random_3d(input, batch_size, input_y, input_x);
    set_to_zero_3d(output, batch_size, output_y, output_x);
    set_to_random_2d(weights, weights_y, weights_x);

    copy_3d_to_1d(input, batch_size, input_y, input_x, input_flat);
    copy_3d_to_1d(output, batch_size, output_y, output_x, output_flat);
    copy_2d_to_1d(weights, weights_y, weights_x, weights_flat);

    copy_3d_to_1d(input, batch_size, input_y, input_x, input_cpu);
    copy_3d_to_1d(output, batch_size, output_y, output_x, output_cpu);
    copy_2d_to_1d(weights, weights_y, weights_x, weights_cpu);

    print_3d(input, batch_size, input_y, input_x);
    //print_3d(output, batch_size, output_y, output_x);
    print_2d(weights, weights_y, weights_x);

    cout << endl << endl << "initializing opencl" << endl;
    initialize_opencl(batch_size, input_y, input_x, weights_y, weights_x, output_y, output_x);

    cout << endl << endl << "Correctness check." << endl;

    cout << "3d: " << endl;
    propagate_forward_3d(batch_size, input, input_y, input_x, weights, weights_y, weights_x, output, output_y, output_x);
    print_3d(output, batch_size, output_y, output_x);

    //cout << "1d: " << endl;

    propagate_forward_1d(batch_size, input_flat, input_y, input_x, weights_flat, weights_y, weights_x, output_flat, output_y, output_x);
    //print_1d(output_flat, batch_size, output_y, output_x);

    /*
    cout << "opencl values on CPU: " << endl;
    for (uint32_t i = 0; i < input_y * input_x; i++) {
        cout << "inputs[" << i << "]: " << input_cpu[i] << endl;
    }

    for (uint32_t i = 0; i < weights_y * weights_x; i++) {
        cout << "weights[" << i << "]: " << weights_cpu[i] << endl;
    }

    for (uint32_t i = 0; i < output_y * output_x; i++) {
        cout << "outputs[" << i << "]: " << output_cpu[i] << endl;
    }

    cout << endl << endl;

    cout << "opencl before: " << endl;
    print_1d(output_cpu, batch_size, output_y, output_x);
    */

    propagate_forward_opencl(input_cpu, weights_cpu, output_cpu);

    cout << "opencl after: " << endl;
    print_1d(output_cpu, batch_size, output_y, output_x);

    cout << "Are 3d and 1d the same? " << check_equal(batch_size, output, output_y, output_x, output_flat) << endl;
    cout << "Are 3d and opencl the same? " << check_equal(batch_size, output, output_y, output_x, output_cpu) << endl;

    //exit(1);

    uint64_t count = 1000;

    cout << endl << endl << "Running " << count << " propagate forwards 3d" << endl;

    using namespace std::chrono;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    for (uint32_t i = 0; i < count; i++) {
        propagate_forward_3d(batch_size, input, input_y, input_x, weights, weights_y, weights_x, output, output_y, output_x);
    }

    high_resolution_clock::time_point t2 = high_resolution_clock::now();

    duration<float, std::milli> time_span = t2 - t1;

    cout << time_span.count() / 1000.0 << " seconds." << endl;

    /*********************************************************************************/

    cout << endl << endl << "Running " << count << " propagate forwards 1d" << endl;

    t1 = high_resolution_clock::now();

    for (uint32_t i = 0; i < count; i++) {
        propagate_forward_1d(batch_size, input_flat, input_y, input_x, weights_flat, weights_y, weights_x, output_flat, output_y, output_x);
    }

    t2 = high_resolution_clock::now();

    time_span = t2 - t1;

    cout << time_span.count() / 1000.0 << " seconds." << endl;

    /*********************************************************************************/

    cout << endl << endl << "Running " << count << " propagate forwards opencl" << endl;

    t1 = high_resolution_clock::now();

    for (uint32_t i = 0; i < count; i++) {
        propagate_forward_opencl(input_cpu, weights_cpu, output_cpu);
    }

    t2 = high_resolution_clock::now();

    time_span = t2 - t1;

    cout << time_span.count() / 1000.0 << " seconds." << endl;

    clReleaseMemObject(input_opencl);
    clReleaseMemObject(output_opencl);
    clReleaseMemObject(weights_opencl);

    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
}


