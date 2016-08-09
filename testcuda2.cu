#include <stdio.h>

////// placeholder for auto-generated code

typedef struct {
  float test;
} Wacky;

typedef Wacky Aggregator;

__host__ __device__ void zero(Aggregator* aggregator) {
  aggregator.test = 0.0f;
}

__host__ __device__ void increment(Aggregator* aggregator, float input_x, float input_y, float weight) {
  aggregator.test += weight * (input_x + input_y);
}

__host__ __device__ void combine(Aggregator* total, Aggregator* item) {
  total.test += item.test;
}

//////

__device__ int threadId() {
  return (blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z) * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
}

extern __shared__ unsigned char sharedMemory[];

__global__ void initialize(int sharedMemoryOffset) {
  Aggregator* threadLocal = (Aggregator*)((size_t)sharedMemory + sharedMemoryOffset + threadId()*sizeof(Aggregator));
  zero(threadLocal);
}

// the input fields are auto-generated
__device__ void fill(int sharedMemoryOffset, float input_x, float input_y, float weight) {
  Aggregator* threadLocal = (Aggregator*)((size_t)sharedMemory + sharedMemoryOffset + threadId()*sizeof(Aggregator));
  increment(threadLocal, input_x, input_y, weight);
}

// the input fields are auto-generated
__global__ void fillAll(int sharedMemoryOffset, float* input_x, float* input_y, float* weight) {
  int id = threadId();
  fill(sharedMemoryOffset, input_x[id], input_y[id], weight[id]);
}

__global__ void extractFromBlock(int sharedMemoryOffset, int numThreadsPerBlock, Aggregator* sumOverBlock) {
  Aggregator *blockLocal = sumOverBlock[blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z];
  zero(sumOverBlock);
  for (int thread = 0;  thread < numThreadsPerBlock;  ++thread) {
    Aggregator* singleAggregator = (Aggregator*)((size_t)sharedMemory + sharedMemoryOffset + thread*sizeof(Aggregator));
    combine(blockLocal, singleAggregator);
  }
}

void extractAll(int sharedMemoryOffset, int numBlocks, int numThreadsPerBlock, Aggregator* sumOverAll) {
  Aggregator *sumOverBlock = NULL;
  cudaMalloc((void**)&sumOverBlock, numBlocks * sizeof(Aggregator));
  extractFromBlock<<<numBlocks, 1, sharedMemoryOffset + numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset, numThreadsPerBlock, sumOverBlock);

  Aggregator *sumOverBlock2 = malloc(numBlocks * sizeof(Aggregator));
  cudaMemcpy(&sumOverBlock2, sumOverBlock, numBlocks * sizeof(Aggregator), cudaMemcpyDeviceToHost);
  cudaFree(sumOverBlock);

  zero(sumOverAll);
  for (int block = 0;  block < numBlocks;  ++block)
    combine(sumOverAll, &sumOverBlock2[block]);

  free(sumOverBlock2);
}

int main(int argc, char** argv) {


}
