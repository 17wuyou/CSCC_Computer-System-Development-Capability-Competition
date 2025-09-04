#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include<sys/time.h>

#define BDIMX 32
#define BDIMY 16
#define IPAD  4

// 定义计时器
struct my_timer
{
    struct timeval start_time, end_time;
    double time_use; // us
    void start(){
		gettimeofday(&start_time, NULL);
    }
    void stop(){
		gettimeofday(&end_time, NULL);
		time_use = (end_time.tv_sec-start_time.tv_sec)*1.0e6 + end_time.tv_usec-start_time.tv_usec;
    }	
};

void printData(char *msg, int *in,  const int size)
{
    printf("%s:", msg);
    int sum;

    for (int i = 0; i < size; i++)
    {
    	sum += in[i];
    }
	
    printf("%5d", sum);
    fflush(stdout);
    
    return;
}

__global__ void setRowReadRow (int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.y][threadIdx.x] ;
}

__global__ void setColReadCol (int *out)
{
    // static shared memory
    __shared__ int tile[BDIMX][BDIMY];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // shared memory store operation
    tile[threadIdx.x][threadIdx.y] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[threadIdx.x][threadIdx.y];
}

__global__ void setRowReadCol(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed coordinate (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow];
}

__global__ void setRowReadColDyn(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * blockDim.x + irow;

    // shared memory store operation
    tile[idx] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[col_idx];
}

__global__ void setRowReadColPad(int *out)
{
    // static shared memory
    __shared__ int tile[BDIMY][BDIMX + IPAD];

    // mapping from 2D thread index to linear memory
    unsigned int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = idx / blockDim.y;
    unsigned int icol = idx % blockDim.y;

    // shared memory store operation
    tile[threadIdx.y][threadIdx.x] = idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[idx] = tile[icol][irow] ;
}

__global__ void setRowReadColDynPad(int *out)
{
    // dynamic shared memory
    extern  __shared__ int tile[];

    // mapping from thread index to global memory index
    unsigned int g_idx = threadIdx.y * blockDim.x + threadIdx.x;

    // convert idx to transposed (row, col)
    unsigned int irow = g_idx / blockDim.y;
    unsigned int icol = g_idx % blockDim.y;

    unsigned int row_idx = threadIdx.y * (blockDim.x + IPAD) + threadIdx.x;

    // convert back to smem idx to access the transposed element
    unsigned int col_idx = icol * (blockDim.x + IPAD) + irow;

    // shared memory store operation
    tile[row_idx] = g_idx;

    // wait for all threads to complete
    __syncthreads();

    // shared memory load operation
    out[g_idx] = tile[col_idx];
}

int main(int argc, char **argv)
{

       	hipDeviceProp_t props;
       	int deviceID = 0;
       	hipGetDeviceProperties(&props, deviceID);
       	printf("%s at ", argv[0]);
       	printf("device %d: %s ", deviceID, props.name);
       	hipSetDevice(deviceID);
        
       	hipSharedMemConfig pConfig;
       	hipDeviceGetSharedMemConfig ( &pConfig );
       	printf("with Bank Mode:%s ", pConfig == 1 ? "4-Byte" : "8-Byte");

       	int nx = BDIMX;
   	   	int ny = BDIMY;

      	 bool iprintf = 0;
       
       	if (argc > 1) iprintf = atoi(argv[1]);
       
      	size_t nBytes = nx * ny * sizeof(int);
       
       	dim3 block (BDIMX, BDIMY);
		dim3 grid  (nx/BDIMX, ny/BDIMY);
    	printf("<<< grid (%d,%d) block (%d,%d)>>>\n", grid.x, grid.y, block.x, block.y);
        
        int *d_C;
    	hipMalloc((int**)&d_C, nBytes);
        int *gpuRef  = (int *)malloc(nBytes);
       	
        hipLaunchKernelGGL(setRowReadRow,grid,block,0,0,d_C);
        hipLaunchKernelGGL(setColReadCol,grid,block,0,0,d_C);
        
        my_timer timer1;
       	hipMemset(d_C, 0, nBytes);
        timer1.start();
        hipLaunchKernelGGL(setRowReadRow,grid,block,0,0,d_C);
        hipDeviceSynchronize();
        timer1.stop();
        hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
        
        printData((char*)"set row read row           ", gpuRef, nx * ny);
        printf("\t cost time : %0.5f ms\n",timer1.time_use/1000);
        
        my_timer timer2;
        hipMemset(d_C, 0, nBytes);
        timer2.start();
        hipLaunchKernelGGL(setColReadCol,grid,block,0,0,d_C);
        hipDeviceSynchronize();
        timer2.stop();
        hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
        
       	printData((char*)"set col read col           ", gpuRef, nx * ny);
        printf("\t cost time : %0.5f ms\n",timer2.time_use/1000);
        
        my_timer timer6;
        hipMemset(d_C, 0, nBytes);
        timer6.start();
        hipLaunchKernelGGL(setRowReadCol,grid,block,0,0,d_C);
        hipDeviceSynchronize();
        timer6.stop();
        hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
        
       	printData((char*)"set row read col           ", gpuRef, nx * ny);
        printf("\t cost time : %0.5f ms\n",timer6.time_use/1000);
		
        my_timer timer3;
		hipMemset(d_C, 0, nBytes);
        timer3.start();
        hipLaunchKernelGGL(setRowReadColDyn,grid,block,BDIMX*BDIMY*sizeof(int),0,d_C);
        hipDeviceSynchronize();
        timer3.stop();
        hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
        
        printData((char*)"set row read col dyn       ", gpuRef, nx * ny);  
        printf("\t cost time : %0.5f ms\n",timer3.time_use/1000);
        
        my_timer timer4;
        hipMemset(d_C, 0, nBytes);
        timer4.start();
        hipLaunchKernelGGL(setRowReadColPad,grid,block,0,0,d_C);
        hipDeviceSynchronize();
        timer4.stop();
        hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
        
       	printData((char*)"set row read col pad       ", gpuRef, nx * ny); 
        printf("\t cost time : %0.5f ms\n",timer4.time_use/1000);
        
        my_timer timer5;
        hipMemset(d_C, 0, nBytes);
        timer5.start();
        hipLaunchKernelGGL(setRowReadColDynPad,grid,block,(BDIMX + IPAD)*BDIMY*sizeof(int),0,d_C);
        hipDeviceSynchronize();
        timer5.stop();
        hipMemcpy(gpuRef, d_C, nBytes, hipMemcpyDeviceToHost);
        
        printData((char*)"set row read col dyn pad   ", gpuRef, nx * ny); 
        printf("\t cost time : %0.5f ms\n",timer5.time_use/1000);
        
        hipFree(d_C);
    	free(gpuRef);

    	// reset device
    	hipDeviceReset();
        
        return 0;
}
