/**
* 实现了二维张量的最大池化操作
--------------------------------
输入（14*14）  
```
6,10,6,2,1,4,0,6,3,1,8,7,5,3,
7,4,9,10,2,0,10,8,5,0,4,6,0,10,
3,10,10,7,10,3,7,9,7,8,2,10,7,10,
4,1,0,10,4,9,7,6,8,6,2,2,7,6,
6,5,5,7,4,4,4,3,6,9,10,2,4,1,
0,10,9,4,0,10,1,2,6,9,7,1,4,9,
3,9,2,10,3,7,6,8,9,8,9,4,5,9,
5,9,8,3,8,7,5,7,4,7,7,8,5,1,
10,7,8,0,6,10,8,9,4,2,4,3,10,3,
5,2,10,8,10,5,0,5,10,6,10,3,0,7,
10,5,6,9,10,4,9,5,3,7,2,8,7,4,
9,4,5,1,7,4,10,6,8,8,0,7,3,9,
0,1,5,8,4,0,6,4,2,2,9,6,7,9,
1,3,3,8,8,8,9,4,0,8,8,8,4,6,
```
池化结果(7*7)为:
```
10,10,4,10,5,8,10,
10,10,10,9,8,10,10,
10,9,10,4,9,10,9,
9,10,8,8,9,9,9,
10,10,10,9,10,10,10,
10,9,10,10,8,8,9,
3,8,8,9,8,9,9,
```
-----------------------------------
*/

#include <stdio.h>

#define W 14
#define H 14
#define TB 7
#define KERNEL_SIZE 2

/**
 * 对卷积结果实现最大池化操作
 * input: 28*28的图片
 * output: 14*14卷积结果(28/2)
 * kernel_size = 2*2
 */
__global__ void MaxPooling2DKernel(float *output, float *input, int inputSize, int kernelSize)
{
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    const int width = inputSize / kernelSize;

    if (col >= inputSize || row >= inputSize ||
        col % kernelSize != 0 || row % kernelSize != 0)
        return;

    int curCol = 0;
    int curRow = 0;
    float maxValue = -100000.0f;

    for (int i = 0; i < kernelSize; ++i)
    {
        for (int j = 0; j < kernelSize; ++j)
        {
            curCol = col + j;
            curRow = row + i;
            if (maxValue < input[curRow * inputSize + curCol])
            {
                maxValue = input[curRow * inputSize + curCol];
            }
        }
    }
    output[row / kernelSize * width + col / kernelSize] = maxValue;
}

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
    int poolOutSize = (W / KERNEL_SIZE) * (W / KERNEL_SIZE);
    int mSize = imgSize * sizeof(float);

    // 1、在Host端分配内存
    float *h_A = (float *)malloc(mSize);
    float *h_C = (float *)malloc(poolOutSize * sizeof(float));

    // 2、初始化A
    for (int i = 0; i < imgSize; ++i)
    {
        h_A[i] = rand_num(0, 10);
    }
    display(h_A, W, H);

    // 3、在Device端为A分配内存
    float *d_A = NULL;
    cudaMalloc(&d_A, mSize);

    // 4、将A从host端拷贝到Device端
    cudaMemcpy(d_A, h_A, mSize, cudaMemcpyHostToDevice);

    // 5、在Device端为C分配内存
    float *d_C = NULL;
    cudaMalloc(&d_C, poolOutSize * sizeof(float));

    // 6、启动kernel计算
    dim3 dimBlock(TB, TB);
    int tb = (W + TB - 1) / TB;
    dim3 dimGrid(tb, tb);
    // Conv2DKernel<<<dimGrid, dimBlock>>>((float(*)[28])d_C, (float(*)[32])d_A, (float(*)[5])d_Kernel);
    MaxPooling2DKernel<<<dimGrid, dimBlock>>>(d_C, d_A, W, KERNEL_SIZE);

    // 7、将计算结果从Device端复制回Host端
    cudaMemcpy(h_C, d_C, poolOutSize * sizeof(float), cudaMemcpyDeviceToHost);

    display(h_C, W / KERNEL_SIZE, W / KERNEL_SIZE);

    // 8、释放内存
    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);

    return 0;
}
