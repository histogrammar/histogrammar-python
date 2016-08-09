#include <stdio.h>

namespace HistogrammarCUDA_1 {

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

  __host__ void toJson(Aggregator* aggregator, FILE* out) {
    fprintf(out, "{\"version\": 123, \"type\": \"wacky\", \"data\": {\"test\": %g}}", aggregator->test);
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
  }

  // the input fields are auto-generated
  __global__ void fillAll(int sharedMemoryOffset, float* input_x, float* input_y, float* weight) {
    int id = threadId() + blockId() * blockSize();
    fill(sharedMemoryOffset, input_x[id], input_y[id], weight[id]);
  }

  __global__ void extractFromBlock(int sharedMemoryOffset, int numThreadsPerBlock, Aggregator* sumOverBlock) {
    Aggregator *blockLocal = &sumOverBlock[blockId()];
    zero(blockLocal);
    for (int thread = 0;  thread < numThreadsPerBlock;  ++thread) {
      Aggregator* singleAggregator = (Aggregator*)((size_t)sharedMemory + sharedMemoryOffset + thread*sizeof(Aggregator));
      combine(blockLocal, singleAggregator);
    }
  }

  void extractAll(int sharedMemoryOffset, int numBlocks, int numThreadsPerBlock, Aggregator* sumOverAll) {
    Aggregator* sumOverBlock = NULL;
    errorCheck(cudaMalloc((void**)&sumOverBlock, numBlocks * sizeof(Aggregator)));

    extractFromBlock<<<numBlocks, 1, sharedMemoryOffset + numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset, numThreadsPerBlock, sumOverBlock);
    errorCheck(cudaPeekAtLastError());
    errorCheck(cudaDeviceSynchronize());

    Aggregator* sumOverBlock2 = (Aggregator*)malloc(numBlocks * sizeof(Aggregator));
    errorCheck(cudaMemcpy(sumOverBlock2, sumOverBlock, numBlocks * sizeof(Aggregator), cudaMemcpyDeviceToHost));
    errorCheck(cudaFree(sumOverBlock));

    zero(sumOverAll);
    for (int block = 0;  block < numBlocks;  ++block)
      combine(sumOverAll, &sumOverBlock2[block]);

    free(sumOverBlock2);
  }

  void test() {
    int numBlocks = 2;
    int numThreadsPerBlock = 5;
    int sharedMemoryOffset = 0;

    initialize<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset);
    errorCheck(cudaPeekAtLastError());
    errorCheck(cudaDeviceSynchronize());

    float x_cpu[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float y_cpu[10] = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f, 2.0f};
    float weight_cpu[10] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    float* x_gpu;
    float* y_gpu;
    float* weight_gpu;

    errorCheck(cudaMalloc((void**)&x_gpu, 10 * sizeof(float)));
    errorCheck(cudaMalloc((void**)&y_gpu, 10 * sizeof(float)));
    errorCheck(cudaMalloc((void**)&weight_gpu, 10 * sizeof(float)));

    errorCheck(cudaMemcpy(x_gpu, x_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));
    errorCheck(cudaMemcpy(y_gpu, y_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));
    errorCheck(cudaMemcpy(weight_gpu, weight_cpu, 10 * sizeof(float), cudaMemcpyHostToDevice));

    fillAll<<<numBlocks, numThreadsPerBlock, numThreadsPerBlock * sizeof(Aggregator)>>>(sharedMemoryOffset, x_gpu, y_gpu, weight_gpu);
    errorCheck(cudaPeekAtLastError());
    errorCheck(cudaDeviceSynchronize());

    errorCheck(cudaFree(x_gpu));
    errorCheck(cudaFree(y_gpu));
    errorCheck(cudaFree(weight_gpu));

    Aggregator sumOverAll;
    extractAll(sharedMemoryOffset, numBlocks, numThreadsPerBlock, &sumOverAll);

    toJson(&sumOverAll, stdout);
    printf("\n");
  }

}

int main(int argc, char** argv) {
  HistogrammarCUDA_1::test();
}
