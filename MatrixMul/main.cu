/**
* 两个矩阵相乘：
* C = A * B
*/

#include <stdio.h>

#define W 512
#define H 512
#define TB 32 

__global__ void MatrixMulKernel(float *C, float *A, float *B, int width)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if ((col >= width) || (row >= width)) return; // avoid illegal access
  float c = 0.0f;
  for (int k = 0; k < width; ++k){
      c += A[row * width + k] * B[k * width + col];
  }
  C[row * width + col] = c;
}

void display(float *Arr, int size){
    printf("size %d\n", size);
    for (int i=0; i < size ; ++i){
        printf("%f\t", Arr[i]);
        if (i % W == 0) printf("\n");
    }
    printf("\n");
}

int main()
{
  int mtsize = W * H;
  int msize = mtsize * sizeof(float);
  
  // 1、在Host端分配内存
  float *A = (float *)malloc(msize);
  float *B = (float *)malloc(msize);
  float *C = (float *)malloc(msize);
  
  // 2、初始化A和B
  for (int i = 0; i < mtsize; ++i)
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
  dim3 dimBlock(TB, TB);
  int tb = (W + TB - 1) / TB;
  dim3 dimGrid(tb, tb);
  MatrixMulKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, W);

  // 7、将计算结果从Device端复制回Host端
  cudaMemcpy(C, d_C, msize, cudaMemcpyDeviceToHost);
  display(C, mtsize);

  // 8、释放内存
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);
  
  return 0;
}
