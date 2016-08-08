#include <stdio.h>

namespace histogrammar {
  typedef struct {
    float values[10];
  } Bn10CtCtCtCt;

  typedef Aggregator Bn10CtCtCtCt;

  void __global__ initialize(Aggregator* aggregator) {
    Aggregator *storage = &aggregator[threadIdx.x];
    for (int i = 0;  i < 10;  ++i)
      storage->values[threadIdx.x].values[i] = 12;
  }

  void __global__ fill(Aggregator* aggregator, float* data, float* weights) {
    float datum = data[blockIdx.x * blockDim.x + threadIdx.x];
    float weight = weights[blockIdx.x * blockDim.x + threadIdx.x];




  }

  Aggregator *allocate(int numThreads) {
    Aggregator *out;
    cudaMalloc((void**)&out, numThreads * sizeof(Aggregator));
    initialize<<<1, numThreads>>>(out);
    return out;
  }




}


namespace histogrammar {
  class Aggregator {
  public:
    



  };


  typedef struct {
    float test;
  } Histogram;

  __global__ void init(Histogram* histogram) {
    const int index = threadIdx.x;
    histogram[index].test = 5;
  }

  void fillall() {
    Histogram* working;
    Histogram output[100];

    cudaMalloc((void**)&working, 100 * sizeof(Histogram));

    histogrammar::init<<<1, 100>>>(working);

    cudaMemcpy(output, working, 100 * sizeof(Histogram), cudaMemcpyDeviceToHost);

    cudaFree(working);

    for (int i = 0;  i < 100;  i++)
      printf("%g\n", output[i].test);
  }
}

int main(int argc, char** argv) {
  histogrammar::fillall();
}
