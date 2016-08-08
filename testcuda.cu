#include <stdio.h>

typedef struct {
  float values[10];
} Bn10CtCtCtCt;

typedef Bn10CtCtCtCt Aggregator;

#define numBlocks 1

#define numThreadsPerBlock 100

#define threadId ((blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z) * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x)

extern __shared__ Aggregator threadLocal[];

void __global__ initialize() {
  Aggregator* mine = &threadLocal[threadId];

  for (int i = 0;  i < 10;  ++i)
    mine->values[i] = 0;
}

void __global__ increment() {
  Aggregator* mine = &threadLocal[threadId];
  for (int i = 0;  i < 10;  ++i)
    mine->values[i] += 1;
}

void __device__ initTotalGPU(Aggregator* total) {
  for (int i = 0;  i < 10;  ++i)
    total->values[i] = 0;
}

void __device__ addToTotalGPU(Aggregator* total, Aggregator* item) {
  for (int i = 0;  i < 10;  ++i)
    total->values[i] += item->values[i];
}

void initTotalCPU(Aggregator* total) {
  for (int i = 0;  i < 10;  ++i)
    total->values[i] = 0;
}

void addToTotalCPU(Aggregator* total, Aggregator* item) {
  for (int i = 0;  i < 10;  ++i)
    total->values[i] += item->values[i];
}

void __global__ extract(Aggregator* globalOnGPU) {
  initTotalGPU(globalOnGPU);
  for (int index = 0;  index < numThreadsPerBlock;  ++index) {
    Aggregator* mine = &threadLocal[index];
    addToTotalGPU(globalOnGPU, mine);
  }
}

int main(int argc, char** argv) {
    initialize<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>();

    increment<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>();
    increment<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>();
    increment<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>();

    Aggregator *globalOnGPU = NULL;
    cudaMalloc((void**)&globalOnGPU, sizeof(Aggregator));

    Aggregator total;
    initTotalCPU(&total);

    for (int block = 0;  block < numBlocks;  ++block) {
      extract<<<1, 1, numThreadsPerBlock * sizeof(Aggregator)>>>(globalOnGPU);

      Aggregator globalOnCPU;
      cudaMemcpy(&globalOnCPU, globalOnGPU, sizeof(Aggregator), cudaMemcpyDeviceToHost);

      addToTotalCPU(&total, &globalOnCPU);
    }

    cudaFree(globalOnGPU);

    for (int i = 0;  i < 10;  i++)
      printf("values[%d] = %g\n", i, total.values[i]);
}
