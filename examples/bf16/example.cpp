#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_bf16.h>

// Macro to check for errors from CUDA driver API calls.
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    CUresult err = call;                                                       \
    if (err != CUDA_SUCCESS) {                                                 \
      const char *errStr;                                                      \
      cuGetErrorString(err, &errStr);                                          \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              errStr);                                                         \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Data tile to be passed to the kernel.
__nv_bfloat16 data[] = {0,   5,   10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,
                65,  70,  75,  80,  85,  90,  95,  100, 105, 110, 115, 120, 125,
                130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190,
                195, 200, 205, 210, 215, 220, 225, 230, 235, 240, 245, 250, 255,
                260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 320,
                325, 330, 335, 340, 345, 350, 355, 360, 365, 370, 375, 380, 385,
                390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450,
                455, 460, 465, 470, 475, 480, 485, 490, 495, 500, 505, 510, 515,
                520, 525, 530, 535, 540, 545, 550, 555, 560, 565, 570, 575, 580,
                585, 590, 595, 600, 605, 610, 615, 620, 625, 630, 635, 
                16,  86,
                31,  96,  6,   39,  79,  11,  100, 117, 118, 37,  106, 128,  11,
                13,  43,  111, 104, 64,  22,  111, 106, 112, 0, 102, 41, 80, 46,
                64,
                112, 50, 90, 18, 14, 41, 114, 16, 41, 31, 82, 31, 34, 89, 82, 15,
                52, 100, 91, 113, 122, 46, 83, 69, 0, 12, 128, 1, 72, 116, 46, 111,
                72, 98, 99, 70, 121, 80, 103, 127, 123, 43, 95, 79, 127, 125, 22,
                123, 1, 5, 0, 43, 6, 62, 86, 104, 117, 51, 14, 53, 46, 114, 88,
                84, 51, 39, 24, 120, 26, 65, 120, 16, 102, 23, 77, 66, 65, 52,
                78, 93, 73, 91, 14, 75, 36, 78, 110, 9, 66, 88, 46, 55, 116, 83,
                44, 108, 19, 116};

int main() {
  // Declare and initialize CUDA driver API handles.
  CUdevice cuDevice;
  CUcontext cuContext;
  CUmodule cuModule;
  CUfunction example_kernel;
  CUstream stream;

  CUDA_CHECK(cuInit(0));
  CUDA_CHECK(cuDeviceGet(&cuDevice, 0));
  CUDA_CHECK(cuCtxCreate(&cuContext, NULL, 0, cuDevice));
  CUDA_CHECK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));

  // Load the compiled cubin file and get the entry CUDA Tile IR function.
  // CUDA Tile IR bytecode can also be directly loaded (JIT compilation).
  CUDA_CHECK(cuModuleLoad(&cuModule, "example.tilebc"));
  CUDA_CHECK(cuModuleGetFunction(&example_kernel, cuModule, "example_kernel"));

  // Allocate memory on the device and copy the input data to it.
  CUdeviceptr data_ptr;
  CUDA_CHECK(cuMemAlloc(&data_ptr, sizeof(data)));
  CUDA_CHECK(cuMemcpyHtoD(data_ptr, data, sizeof(data)));

  CUdeviceptr out_ptr;
  CUDA_CHECK(cuMemAlloc(&out_ptr, sizeof(data)));

  int size = 16;

  // Launch the kernel.
  void *kernel_args[] = {&data_ptr, &out_ptr, &size };
  CUDA_CHECK(cuLaunchKernel(example_kernel, // function
                            4, 4, 1,        // grid dims: must be (1,1,1)
                            1, 1, 1,        // block dims
                            0,              // shared memory bytes: must be 0
                            stream,         // cuda stream
                            kernel_args,    // kernel arguments
                            NULL            // extra parameters
                            ));

  CUDA_CHECK(cuCtxSynchronize());
  __nv_bfloat16 host_out[sizeof(data) / sizeof(__nv_bfloat16)];
  CUDA_CHECK(cuMemcpyDtoH(host_out, out_ptr, sizeof(data)));
  // print the output data.
  for (size_t i = 0; i < sizeof(data) / sizeof(__nv_bfloat16); ++i) {
    printf("out[%zu] = %f\n", i, __bfloat162float(host_out[i]));
  }
  // Clean up.
  CUDA_CHECK(cuModuleUnload(cuModule));
  CUDA_CHECK(cuCtxDestroy(cuContext));

  return 0;
}
