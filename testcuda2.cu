#include <stdio.h>

////// placeholder for auto-generated code

typedef struct {
  float test;
} Wacky;

typedef Wacky Aggregator;

__host__ __device__ void zero(Aggregator* aggregator) {
  aggregator->test = 0.0f;
}

__host__ __device__ void increment(Aggregator* aggregator, float input_x, float input_y, float weight) {
  aggregator->test += weight * (input_x + input_y);
}

__host__ __device__ void combine(Aggregator* total, Aggregator* item) {
  total->test += item->test;
}

//////

__device__ int blockId() {
  return blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
}

__device__ int blockSize() {
  return blockDim.x * blockDim.y * blockDim.z;
}

__device__ int threadId() {
  return threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
}

extern __shared__ unsigned char sharedMemory[];

void errorCheck(cudaError_t code) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(code));
    exit(code);
  }
}

__global__ void initialize(int sharedMemoryOffset) {
  Aggregator* threadLocal = (Aggregator*)((size_t)sharedMemory + sharedMemoryOffset + threadId()*sizeof(Aggregator));
  zero(threadLocal);
}

// the input fields are auto-generated
__device__ void fill(int sharedMemoryOffset, float input_x, float input_y, float weight) {
  Aggregator* threadLocal = (Aggregator*)((size_t)sharedMemory + sharedMemoryOffset + threadId()*sizeof(Aggregator));
  increment(threadLocal, input_x, input_y, weight);

  printf("    fill %ld %g\n", (size_t)threadLocal, threadLocal->test);
}

// the input fields are auto-generated
__global__ void fillAll(int sharedMemoryOffset, float* input_x, float* input_y, float* weight) {
  printf("here %g %g %g\n", input_x[0], input_y[0], weight[0]);

  int id = threadId() + blockId() * blockSize();
  fill(sharedMemoryOffset, input_x[id], input_y[id], weight[id]);
}

__global__ void extractFromBlock(int sharedMemoryOffset, int numThreadsPerBlock, Aggregator* sumOverBlock) {
  Aggregator *blockLocal = &sumOverBlock[blockId()];
  zero(blockLocal);
  for (int thread = 0;  thread < numThreadsPerBlock;  ++thread) {
    Aggregator* singleAggregator = (Aggregator*)((size_t)sharedMemory + sharedMemoryOffset + thread*sizeof(Aggregator));

    printf("    extract %ld %g\n", (size_t)singleAggregator, singleAggregator->test);

    combine(blockLocal, singleAggregator);
  }
}

void extractAll(int sharedMemoryOffset, int numBlocks, int numThreadsPerBlock, Aggregator* sumOverAll) {
  printf("eight\n");

  Aggregator* sumOverBlock = NULL;
  errorCheck(cudaMalloc((void**)&sumOverBlock, numBlocks * sizeof(Aggregator)));

  printf("nine\n");

  extractFromBlock<<<numBlocks, 1, sharedMemoryOffset + numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset, numThreadsPerBlock, sumOverBlock);
  errorCheck(cudaPeekAtLastError());
  errorCheck(cudaDeviceSynchronize());

  printf("ten\n");

  Aggregator* sumOverBlock2 = (Aggregator*)malloc(numBlocks * sizeof(Aggregator));

  printf("eleven\n");

  errorCheck(cudaMemcpy(sumOverBlock2, sumOverBlock, numBlocks * sizeof(Aggregator), cudaMemcpyDeviceToHost));

  printf("twelve\n");

  errorCheck(cudaFree(sumOverBlock));

  printf("thirteen\n");

  zero(sumOverAll);

  printf("fourteen\n");

  for (int block = 0;  block < numBlocks;  ++block) {
    printf("    block %d %g\n", block, sumOverBlock2[block].test);

    combine(sumOverAll, &sumOverBlock2[block]);
  }

  printf("fifteen\n");

  free(sumOverBlock2);

  printf("sixteen\n");
}

int main(int argc, char** argv) {
  int numBlocks = 3;
  int numThreadsPerBlock = 2;
  int sharedMemoryOffset = 4;

  printf("one\n");

  initialize<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset);
  errorCheck(cudaPeekAtLastError());
  errorCheck(cudaDeviceSynchronize());

  printf("two\n");

  float x_cpu[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  float y_cpu[10] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  float weight_cpu[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  float* x_gpu;
  float* y_gpu;
  float* weight_gpu;

  printf("three\n");

  errorCheck(cudaMalloc((void**)&x_gpu, 10 * sizeof(float)));
  errorCheck(cudaMalloc((void**)&y_gpu, 10 * sizeof(float)));
  errorCheck(cudaMalloc((void**)&weight_gpu, 10 * sizeof(float)));

  printf("four\n");

  errorCheck(cudaMemcpy(x_gpu, x_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));
  errorCheck(cudaMemcpy(y_gpu, y_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));
  errorCheck(cudaMemcpy(weight_gpu, weight_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));

  printf("five\n");

  fillAll<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset, x_gpu, y_gpu, weight_gpu);
  errorCheck(cudaPeekAtLastError());
  errorCheck(cudaDeviceSynchronize());

  printf("six\n");

  errorCheck(cudaFree(x_gpu));
  errorCheck(cudaFree(y_gpu));
  errorCheck(cudaFree(weight_gpu));

  printf("seven\n");

  Aggregator sumOverAll;
  extractAll(sharedMemoryOffset, numBlocks, numThreadsPerBlock, &sumOverAll);

  printf("done %g\n", sumOverAll.test);
}
