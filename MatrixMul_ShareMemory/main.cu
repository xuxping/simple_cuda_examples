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
  // 声明共享内存, TILE_WIDTH + 1 是为防止bank conflict
  // 做的memory padding操作
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH + 1]; 
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH + 1]; 
  
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x; // 0 ~ TILE_WIDTH-1
  int ty = threadIdx.y; // 0 ~ TILE_WIDTH-1

  int row = by * TILE_WIDTH + ty;
  int col = bx * TILE_WIDTH + tx;
  float c = 0.0f;

  if ((col >= width) || (row >= width)) return;  // 避免非法访问
  
  for (int m = 0; m < width / TILE_WIDTH; ++m){
      // 将Global Memory的数据填充到Share Memory中
      Mds[ty][tx] = A[row * width + (m * TILE_WIDTH + tx)];
      Nds[ty][tx] = B[col + (m * TILE_WIDTH + ty) * width];
      // 对线程块中的所有线程进行同步，保证执行完前面的语句后，才会执行下一条语句
      __syncthreads();
    
      for (int k = 0; k < TILE_WIDTH; ++k){
         c += Mds[ty][k] * Nds[k][tx];
      }
      // 在下一次循环开始时，确保当前中的所有线程的更新操作都已经完成
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
  dim3 dimGrid(W / TILE_WIDTH, H / TILE_WIDTH);
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
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
