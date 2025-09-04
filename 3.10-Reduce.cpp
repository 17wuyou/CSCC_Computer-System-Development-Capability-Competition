#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

int recursiveReduce(int *data, int const size)
{
    // terminate check
    if (size == 1) return data[0];

    // renew the stride
    int const stride = size / 2;

    // in-place reduction
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    // call recursively
    return recursiveReduce(data, stride);
}

// Neighbored Pair Implementation with divergence
__global__ void reduceNeighbored (int *g_idata, int *g_odata, unsigned int n)//并行规约分化
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if (idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess (int *g_idata, int *g_odata,
                                      unsigned int n)
//改善并行规约优化
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        // convert tid into local array index
        int index = 2 * stride * tid;

        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}

// Interleaved Pair Implementation with less divergence
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n)
//交错规约
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;

    // boundary check
    if(idx >= n) return;

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}



int main(int argc, char **argv)
{

    // set up device
    int dev = 0;
    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size


    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    hipEvent_t start_time,stop_time;

    hipEventCreate(&start_time); 
    hipEventCreate(&stop_time);
    
    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);
    float time_elapsed  = 1.0f  ;
    

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    hipMalloc((void **) &d_idata, bytes);
    hipMalloc((void **) &d_odata, grid.x * sizeof(int));

    // cpu reduction
    hipEventRecord(start_time,NULL);
    int cpu_sum = recursiveReduce (tmp, size);
    hipEventRecord(stop_time,NULL);
    hipEventElapsedTime(&time_elapsed, start_time,stop_time);
    printf("cpu reduce elapsed %f sec cpu_sum: %d\n", time_elapsed, cpu_sum);

    // kernel 1: reduceNeighbored
    hipMemcpy(d_idata, h_idata, bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    hipEventRecord(start_time,NULL);
    hipLaunchKernelGGL(reduceNeighbored,grid,block,0,0,d_idata,d_odata,size);
    hipDeviceSynchronize();
    hipEventRecord(stop_time,NULL);
    hipEventElapsedTime(&time_elapsed, start_time,stop_time);
    hipMemcpy(h_odata, d_odata, grid.x * sizeof(int),hipMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", time_elapsed, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    hipMemcpy(d_idata, h_idata, bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    hipEventRecord(start_time,NULL);
    hipLaunchKernelGGL(reduceNeighboredLess,grid,block,0,0,d_idata,d_odata,size);
    hipDeviceSynchronize();
    hipEventRecord(stop_time,NULL);
    hipEventElapsedTime(&time_elapsed, start_time,stop_time);
    hipMemcpy(h_odata, d_odata, grid.x * sizeof(int),hipMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", time_elapsed, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    hipMemcpy(d_idata, h_idata, bytes, hipMemcpyHostToDevice);
    hipDeviceSynchronize();
    hipEventRecord(start_time,NULL);
    hipLaunchKernelGGL(reduceInterleaved,grid,block,0,0,d_idata,d_odata,size);
    hipDeviceSynchronize();
    hipEventRecord(stop_time,NULL);
    hipEventElapsedTime(&time_elapsed, start_time,stop_time);
    hipMemcpy(h_odata, d_odata, grid.x * sizeof(int),hipMemcpyDeviceToHost);
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", time_elapsed, gpu_sum, grid.x, block.x);

    

    hipEventDestroy(start_time);
    hipEventDestroy(stop_time);
    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    hipFree(d_idata);
    hipFree(d_odata);

    // reset device
    hipDeviceReset();

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return 0;
}
