#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

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
float data[128] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1};

float data1[128] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1, 1, 1, 1, 1, 1, 
                 1, 1, 1};

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

  CUdeviceptr data_ptr1;
  CUDA_CHECK(cuMemAlloc(&data_ptr1, sizeof(data1)));
  CUDA_CHECK(cuMemcpyHtoD(data_ptr1, data1, sizeof(data1)));

  int M = 8;
  int N = 8;
  int K = 16;

  CUdeviceptr out_ptr;
  CUDA_CHECK(cuMemAlloc(&out_ptr, M * N * sizeof(float)));

  int stride_am = M;  // A is KxM
  int stride_bn = K;  // B is NxK
  int stride_cm = N;  // C is MxN
  // Launch the kernel.
  void *kernel_args[] = {&data_ptr, &data_ptr1, &out_ptr, &M, &N, &K, &stride_am, &stride_bn, &stride_cm};
  CUDA_CHECK(cuLaunchKernel(example_kernel, // function
                            4, 2, 1,        // grid dims: must be (1,1,1)
                            1, 1, 1,        // block dims
                            0,              // shared memory bytes: must be 0
                            stream,         // cuda stream
                            kernel_args,    // kernel arguments
                            NULL            // extra parameters
                            ));

  CUDA_CHECK(cuCtxSynchronize());
  float host_out[M * N];
  CUDA_CHECK(cuMemcpyDtoH(host_out, out_ptr, M * N * sizeof(float)));
  // print the output data.
  for (size_t i = 0; i < M * N; ++i) {
    printf("out[%zu] = %f\n", i, host_out[i]);
  }
  // Clean up.
  CUDA_CHECK(cuModuleUnload(cuModule));
  CUDA_CHECK(cuCtxDestroy(cuContext));

  return 0;
}

