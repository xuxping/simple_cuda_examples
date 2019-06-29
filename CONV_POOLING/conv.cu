/**
* 实现了一张图像的卷积操作
-------------------
输入（5*5）
```
6,10,6,2,1,
4,0,6,3,1,
8,7,5,3,7,
4,9,10,2,0,
10,8,5,0,4,
```
卷积核  
```
2,2,
1,1,
```
输出结果为:
```
36,38,25,10,
23,24,26,18,
43,43,28,22,
44,51,29,8,
```
------------------
*/

#include <stdio.h>

#define W 5
#define H 5
#define TB 2
#define KERNEL_SIZE 2

/**
 * 二维卷积操作
 */
__global__ void Conv2DKernel(float *output, float *input, float *kernel, int inputSize, int kernelSize)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    // 边界检查, 0 ~ inputSize - kernelSize + 1
    const int limit = inputSize - kernelSize + 1;
    if (col >= limit || row >= limit)
        return;

    int curCol = 0;
    int curRow = 0;
    float sum = 0.0f;
    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            curCol = col + j;
            curRow = row + i;
            sum += (kernel[i * kernelSize + j] * input[curRow * inputSize + curCol]);
        }
    }
    output[row * limit + col] = sum;
}

/**
 * 对一张图片实现卷积操作
 * input: 32*32的图片
 * output: 28*28卷积结果(32-5+1=28)
 * kernel_size = 5*5
 */
// __global__ void Conv2DKernel(float output[28][28], float input[32][32], float kernel[5][5])
// {
//     int col = threadIdx.x + blockDim.x * blockIdx.x;
//     int row = threadIdx.y + blockDim.y * blockIdx.y;
//     // const int stride = 1; // 步长为1
//     const int kernel_size = 5;

//     // 边界检查, 0 ~ H - kernel_size + 1
//     if (col >= 28 || row >= 28)
//         return;

//     int curCol = 0;
//     int curRow = 0;
//     float sum = 0.0f;
//     for (int i = 0; i < kernel_size; ++i)
//     {
//         for (int j = 0; j < kernel_size; ++j)
//         {
//             curCol = col + j;
//             curRow = row + i;
//             sum += (kernel[i][j] * input[curRow][curCol]);
//         }
//     }
//     output[row][col] = sum;
// }

void display(float *arr, int w, int h)
{
    for (int i = 0; i < w; ++i)
    {
        for (int j = 0; j < h; ++j)
        {
            printf("%d,", int(arr[i * w + j]));
        }
        printf("\n");
    }
    printf("\n");
}

int rand_num(int start, int end)
{
    return rand() % (end + 1 - start) + start;
}

int main()
{
    int imgSize = W * H;
    int convOutW = W - KERNEL_SIZE + 1;
    int convOutSize = convOutW * convOutW;
    int mSize = imgSize * sizeof(float);

    // 1、在Host端分配内存
    float *h_A = (float *)malloc(mSize);
    float *h_Kernel = (float *)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float *h_C = (float *)malloc(convOutSize * sizeof(float));

    // 2、初始化A和kernel
    for (int i = 0; i < imgSize; ++i)
    {
        h_A[i] = rand_num(0, 10);
    }

    for (int j = 0; j < KERNEL_SIZE * KERNEL_SIZE; ++j)
    {
        h_Kernel[j] = rand_num(0, 2);
    }

    display(h_A, W, H);
    display(h_Kernel, KERNEL_SIZE, KERNEL_SIZE);

    // 3、在Device端为A和B分配内存
    float *d_A = NULL;
    float *d_Kernel = NULL;
    cudaMalloc(&d_A, mSize);
    cudaMalloc(&d_Kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // 4、将A和Kernel从host端拷贝到Device端
    cudaMemcpy(d_A, h_A, mSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kernel, h_Kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // 5、在Device端为C分配内存
    float *d_C = NULL;
    cudaMalloc(&d_C, convOutSize * sizeof(float));

    // 6、启动kernel计算
    dim3 dimBlock(TB, TB);
    int tb = (W + TB - 1) / TB;
    dim3 dimGrid(tb, tb);
    // Conv2DKernel<<<dimGrid, dimBlock>>>((float(*)[28])d_C, (float(*)[32])d_A, (float(*)[5])d_Kernel);
    Conv2DKernel<<<dimGrid, dimBlock>>>(d_C, d_A, d_Kernel, W, KERNEL_SIZE);

    // 7、将计算结果从Device端复制回Host端
    cudaMemcpy(h_C, d_C, convOutSize * sizeof(float), cudaMemcpyDeviceToHost);

    display(h_C, convOutW, convOutW);

    // 8、释放内存
    cudaFree(d_A);
    cudaFree(d_Kernel);
    cudaFree(d_C);
    free(h_A);
    free(h_Kernel);
    free(h_C);

    return 0;
}
