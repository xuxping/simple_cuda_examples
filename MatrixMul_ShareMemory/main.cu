/**
* 使用共享内存Shared Memory实现两个矩阵相乘：
* C = A * B
*/


#include <stdio.h>

#define W 512
#define H 512
#define TILE_WIDTH 4
#define TB 32 

__global__ void MatrixMulKernel(float *C, float *A, float *B, int width)
{
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];  // for A
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];  // for B
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float c = 0.0f;

  if ((col >= width) || (row >= width)) return;  
  
  for (int m = 0; m < width / TILE_WIDTH; ++m){
      Mds[ty][tx] = A[row * width + (m * TILE_WIDTH + tx)];
      Nds[ty][tx] = B[col + (m * TILE_WIDTH + ty) * width];
      // wait all block threads
      __syncthreads();
    
      for (int k = 0; k < TILE_WIDTH; ++k){
         c += Mds[ty][k] * Nds[k][tx];
      }
      __syncthreads();
  }
      C[row*width + col] = c;
}

void display(float *Arr, int size){
    for (int i=0; i < size ; ++i){
        printf("%d\t", int(Arr[i]));
        if (i % W == 0){
           printf("\n");
        }
    }
    printf("\n");

}

int main()
{
  int mtsize = W*H;
  int msize = mtsize *sizeof(float);
  float *A = (float *)malloc(msize);
  float *B = (float *)malloc(msize);
  float *C = (float *)malloc(msize);
 
  for (int i = 0; i < mtsize; ++i)
  {
        A[i] = 1.0f;
        B[i] = 2.0f;
  } 
  printf("display C:...");
  //display(A, mtsize);
  //display(B, mtsize);
  //display(C, mtsize);

  // initial verctor a and b in cuda
  float *d_A = NULL; 
  float *d_B = NULL; 
  cudaMalloc(&d_A, msize);
  cudaMalloc(&d_B, msize);
 
  // copy to A,B to cuda
  cudaMemcpy(d_A, A, msize, cudaMemcpyHostToDevice); 
  cudaMemcpy(d_B, B, msize, cudaMemcpyHostToDevice); 
  
  float *d_C=NULL; 
  cudaMalloc(&d_C, msize);
  dim3 dimGrid(W / TILE_WIDTH, H / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  
  MatrixMulKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_B, W);

  // Copy results from device to host.
  cudaMemcpy(C, d_C, msize, cudaMemcpyDeviceToHost);
  display(C, mtsize);

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  free(A);
  free(B);
  free(C);
  return 0;
}
