#include <iostream>
#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>

#define N 10000

__global__ void add(float *d_A, float *d_B, float *d_C) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        d_C[tid] = d_A[tid] + d_B[tid];
    }
}

int main() {
    //申请数据空间
    float *A = (float *) malloc(N * sizeof(float));
    float *B = (float *) malloc(N * sizeof(float));
    float *C = (float *) malloc(N * sizeof(float));
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    hipMalloc((void **) &d_A, N * sizeof(float));
    hipMalloc((void **) &d_B, N * sizeof(float));
    hipMalloc((void **) &d_C, N * sizeof(float));
    //数据初始化
    for (int i = 0; i < N; i++) {
        A[i] = 1;
        B[i] = 1;
        C[i] = 0;
    }
    hipMemcpy(d_A, A, sizeof(float) * N, hipMemcpyHostToDevice);
    hipMemcpy(d_B, B, sizeof(float) * N, hipMemcpyHostToDevice);
    hipMemcpy(d_C, C, sizeof(float) * N, hipMemcpyHostToDevice);
    dim3 blocksize(256, 1);
    dim3 gridsize(N / 256 + 1, 1);
    // 进行数组相加
    add<<<gridsize, blocksize >>> (d_A, d_B, d_C);
    //结果验证
    hipMemcpy(C, d_C, sizeof(float) * N, hipMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        std::cout << C[i] << std::endl;
    }
    //释放申请空间
    free(A);
    free(B);
    free(C);
    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_C);
}