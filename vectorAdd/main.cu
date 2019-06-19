/**
* 向量相加：
* C = A + B
*/

#include <stdio.h>
#define MATRIX_SIZE 512
#define TB 32 

__global__ void addKernel(float *C, float *A, float *B, int vecLen)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < vecLen){
    C[i] = A[i] + B[i];
  }
}

void display(float *Arr, int size){
    for (int i=0; i < size ; ++i){
        printf("%d\t", int(Arr[i]));
    }
    printf("\n");
}

int main()
{
  // malloc memory in host
  float *A = (float *)malloc(MATRIX_SIZE*sizeof(float));
  float *B = (float *)malloc(MATRIX_SIZE*sizeof(float));
  float *C = (float *)malloc(MATRIX_SIZE*sizeof(float));
 
  for (int i = 0; i < MATRIX_SIZE; ++i)
  {
        A[i] = 1.0f;
        B[i] = 2.0f;
  }

  // initial verctor a and b in cuda
  float *d_A = NULL; 
  float *d_B = NULL; 
  cudaMalloc(&d_A, MATRIX_SIZE*sizeof(float));
  cudaMalloc(&d_B, MATRIX_SIZE*sizeof(float));
 
  // copy to A,B to cuda
  cudaMemcpy(d_A, A, MATRIX_SIZE*sizeof(float), cudaMemcpyHostToDevice); 
  cudaMemcpy(d_B, B, MATRIX_SIZE*sizeof(float), cudaMemcpyHostToDevice); 
  
  float *d_C=NULL; 
  cudaMalloc(&d_C, MATRIX_SIZE*sizeof(float));
  int grid = (MATRIX_SIZE + TB - 1) / TB;
  addKernel<<<grid, TB>>>(d_C, d_A, d_B, MATRIX_SIZE);

  // Copy results to host.
  cudaMemcpy(C, d_C, MATRIX_SIZE*sizeof(float), cudaMemcpyDeviceToHost);
  display(C, MATRIX_SIZE);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);
  return 0;
}
