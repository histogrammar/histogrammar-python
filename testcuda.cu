#include <stdio.h>

typedef struct {
  float values[10];
} Bn10CtCtCtCt;

typedef Bn10CtCtCtCt Aggregator;

extern __shared__ Aggregator threadLocal[];

void __global__ initialize() {
  Aggregator* mine = &threadLocal[threadIdx.x + blockIdx.x*blockDim.x];

  for (int i = 0;  i < 10;  ++i)
    mine->values[i] = 0;
}

void __global__ increment() {
  Aggregator* mine = &threadLocal[threadIdx.x + blockIdx.x*blockDim.x];
  for (int i = 0;  i < 10;  ++i)
    mine->values[i] += 1;
}

void __global__ extract(Aggregator* globalOnGPU) {
  for (int i = 0;  i < 10;  ++i)
    globalOnGPU->values[i] = 0;

  for (int index = 0;  index < 100;  ++index) {
    Aggregator* mine = &threadLocal[index];

    for (int i = 0;  i < 10;  ++i)
      globalOnGPU->values[i] += mine->values[i];
  }
}

int main(int argc, char** argv) {
    Aggregator *globalOnGPU = NULL;
    Aggregator globalOnCPU;
  
    initialize<<<1, 100, 100 * sizeof(Aggregator)>>>();

    increment<<<1, 100, 100 * sizeof(Aggregator)>>>();
    increment<<<1, 100, 100 * sizeof(Aggregator)>>>();
    increment<<<1, 100, 100 * sizeof(Aggregator)>>>();

    cudaMalloc((void**)&globalOnGPU, sizeof(Aggregator));

    extract<<<1, 1, 100 * sizeof(Aggregator)>>>(globalOnGPU);

    cudaMemcpy(&globalOnCPU, globalOnGPU, sizeof(Aggregator), cudaMemcpyDeviceToHost);

    cudaFree(globalOnGPU);

    for (int i = 0;  i < 10;  i++)
      printf("values[%d] = %g\n", i, globalOnCPU.values[i]);
}
