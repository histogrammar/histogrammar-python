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

#define cudaErrorCheck(code) { cudaAssert(code, __FILE__, __LINE__); }

void cudaAssert(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "cudaAssert: %s at %s:%d\n", cudaGetErrorString(code), file, line);
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
  cudaErrorCheck(cudaMalloc((void**)&sumOverBlock, numBlocks * sizeof(Aggregator)));

  printf("nine\n");

  extractFromBlock<<<numBlocks, 1, sharedMemoryOffset + numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset, numThreadsPerBlock, sumOverBlock);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());

  printf("ten\n");

  Aggregator* sumOverBlock2 = (Aggregator*)malloc(numBlocks * sizeof(Aggregator));

  printf("eleven\n");

  cudaErrorCheck(cudaMemcpy(sumOverBlock2, sumOverBlock, numBlocks * sizeof(Aggregator), cudaMemcpyDeviceToHost));

  printf("twelve\n");

  cudaErrorCheck(cudaFree(sumOverBlock));

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
  int numBlocks = 1;
  int numThreadsPerBlock = 1;
  int sharedMemoryOffset = 0;

  printf("one\n");

  initialize<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());

  printf("two\n");

  float x_cpu[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
  float y_cpu[10] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
  float weight_cpu[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

  float* x_gpu;
  float* y_gpu;
  float* weight_gpu;

  printf("three\n");

  cudaErrorCheck(cudaMalloc((void**)&x_gpu, 10 * sizeof(float)));
  cudaErrorCheck(cudaMalloc((void**)&y_gpu, 10 * sizeof(float)));
  cudaErrorCheck(cudaMalloc((void**)&weight_gpu, 10 * sizeof(float)));

  printf("four\n");

  cudaErrorCheck(cudaMemcpy(x_gpu, x_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(y_gpu, y_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));
  cudaErrorCheck(cudaMemcpy(weight_gpu, weight_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));

  printf("five\n");

  fillAll<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset, x_gpu, y_gpu, weight_gpu);
  cudaErrorCheck(cudaPeekAtLastError());
  cudaErrorCheck(cudaDeviceSynchronize());

  printf("six\n");

  cudaErrorCheck(cudaFree(x_gpu));
  cudaErrorCheck(cudaFree(y_gpu));
  cudaErrorCheck(cudaFree(weight_gpu));

  printf("seven\n");

  Aggregator sumOverAll;
  extractAll(sharedMemoryOffset, numBlocks, numThreadsPerBlock, &sumOverAll);

  printf("done %g\n", sumOverAll.test);
}
