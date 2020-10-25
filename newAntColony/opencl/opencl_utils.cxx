#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <math.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void check_error(cl_int err, const char* fmt, ...) {
    va_list argp;
    va_start(argp, fmt);

    if (err < 0) {
        vfprintf(stderr, fmt, argp);
        fprintf(stderr, "\n");
        exit(1);   
    };
}


/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device() {
    cl_platform_id platform;
    cl_device_id dev;
    int err;

    /* Identify a platform */
    err = clGetPlatformIDs(1, &platform, NULL);
    check_error(err, "Couldn't identify a platform");

    /* Access a device */
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
    if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
    }

    check_error(err, "Couldn't access any devices");


    char deviceName[1024];              //this string will hold the devices name
    char vendor[1024];                  //this strirng will hold a platforms vendor

    cl_uint numberOfCores;              //this variable holds the number of cores of on a device
    cl_long amountOfMemory;             //this variable holds the amount of memory on a device
    cl_uint clockFreq;                  //this variable holds the clock frequency of a device
    cl_ulong maxAlocatableMem;          //this variable holds the maximum allocatable memory
    cl_ulong localMem;                  //this variable holds local memory for a device
    cl_bool available;                  //this variable holds if the device is available
    cl_ulong constantBufferSize;
    cl_uint maxWorkItemDimensions;
    size_t maxWorkItemSizes[3];
    size_t maxWorkGroupSize;

    //scan in device information
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(numberOfCores), &numberOfCores, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(amountOfMemory), &amountOfMemory, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(clockFreq), &clockFreq, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxAlocatableMem), &maxAlocatableMem, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMem), &localMem, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_AVAILABLE, sizeof(available), &available, NULL);

    clGetDeviceInfo(dev, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(constantBufferSize), &constantBufferSize, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);

    clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxWorkItemSizes), &maxWorkItemSizes, NULL);

    //print out device information
    printf("\tDevice:\n");
    printf("\t\tName:\t\t\t\t%s\n", deviceName);
    printf("\t\tVendor:\t\t\t\t%s\n", vendor);
    printf("\t\tAvailable:\t\t\t%s\n", available ? "Yes" : "No");
    printf("\t\tCompute Units:\t\t\t%u\n", numberOfCores);
    printf("\t\tClock Frequency:\t\t%u mHz\n", clockFreq);
    printf("\t\tGlobal Memory:\t\t\t%0.00f mb\n", (double)amountOfMemory/1048576);
    printf("\t\tMax Allocateable Memory:\t%0.00f mb\n", (double)maxAlocatableMem/1048576);
    printf("\t\tLocal Memory:\t\t\t%u kb\n\n", (unsigned int)localMem);
    printf("\t\tMax Constant Buffer Size:\t%0.00f mb\n\n", (double)constantBufferSize/1048576);
    printf("\t\tMax Work Group Size:\t\t%u\n", maxWorkGroupSize);
    printf("\t\tMax Work Item Dimensions:\t%u\n", maxWorkItemDimensions);
    printf("\t\tMax Work Item Sizes:\t\t%d %d %d\n", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);



    return dev;
}

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename) {

    cl_program program;
    FILE *program_handle;
    char *program_buffer, *program_log;
    size_t program_size, log_size;
    int err;

    /* Read program file and place content into buffer */
    program_handle = fopen(filename, "r");
    if (program_handle == NULL) {
        fprintf(stderr, "Couldn't find the program file: '%s'", filename);
        exit(1);
    }

    fseek(program_handle, 0, SEEK_END);
    program_size = ftell(program_handle);
    rewind(program_handle);
    program_buffer = (char*)malloc(program_size + 1);
    program_buffer[program_size] = '\0';
    fread(program_buffer, sizeof(char), program_size, program_handle);
    fclose(program_handle);

    /* Create program from file */
    program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer, &program_size, &err);
    check_error(err, "Couldn't create the program: '%s'", filename);
    free(program_buffer);

    /* Build program */
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if(err < 0) {

        /* Find size of log and print to std output */
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        program_log = (char*) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
    }

    return program;
}


