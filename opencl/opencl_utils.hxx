#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif


/* check for an error and print an error message if found */
void check_error(cl_int err, const char* fmt, ...);
 
/* Find a GPU or CPU associated with the first available platform */
cl_device_id create_device();

/* Create program from a file and compile it */
cl_program build_program(cl_context ctx, cl_device_id dev, const char* filename);


