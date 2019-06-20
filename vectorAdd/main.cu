/**
* 向量相加：
* C = A + B
*/

#include <stdio.h>

#define VECTOR_SIZE 512
#define TB 32 

__global__ void addKernel(float *C, float *A, float *B, int vecLen)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < vecLen){ // avoid illegal access
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
  // 1、在Host端分配内存
  int msize = VECTOR_SIZE * sizeof(float);
  float *A = (float *)malloc(msize);
  float *B = (float *)malloc(msize);
  float *C = (float *)malloc(msize);
 
  // 2、初始化A和B
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
        A[i] = 1.0f;
        B[i] = 2.0f;
  }

  // 3、在Device端为A和B分配内存
  float *d_A = NULL; 
  float *d_B = NULL; 
  cudaMalloc(&d_A, msize);
  cudaMalloc(&d_B, msize);
 
  // 4、将A和B从host端拷贝到Device端
  cudaMemcpy(d_A, A, msize, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_B, B, msize, cudaMemcpyHostToDevice); 
  
  // 5、在Device端为C分配内存
  float *d_C = NULL; 
  cudaMalloc(&d_C, msize);

  // 6、启动kernel计算
  int grid = (VECTOR_SIZE + TB - 1) / TB;
  addKernel<<<grid, TB>>>(d_C, d_A, d_B, VECTOR_SIZE);

  // 7、将计算结果从Device端复制回Host端
  cudaMemcpy(C, d_C, msize, cudaMemcpyDeviceToHost);
  display(C, VECTOR_SIZE);

  // 8、释放内存
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);

  return 0;
}
