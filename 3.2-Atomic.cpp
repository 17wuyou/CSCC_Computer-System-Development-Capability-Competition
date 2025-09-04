#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include<sys/time.h>

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

__device__ int myAtomicAdd(int *address, int incr)
{
    // Create an initial guess for the value stored at *address.
    int guess = *address;
    int oldValue = atomicCAS(address, guess, guess + incr);

    // Loop while the guess is incorrect.
    while (oldValue != guess)
    {
        guess = oldValue;
        oldValue = atomicCAS(address, guess, guess + incr);
    }

    return oldValue;
}

__global__ void atomics(int *shared_var, int *values_read, int N, int iters)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    values_read[tid] = atomicAdd(shared_var, 1);

    for (i = 0; i < iters; i++)
    {
        atomicAdd(shared_var, 1);
    }
}

__global__ void unsafe(int *shared_var, int *values_read, int N, int iters)
{
    int i;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid >= N) return;

    int old = *shared_var;
    *shared_var = old + 1;
    values_read[tid] = old;

    for (i = 0; i < iters; i++)
    {
        int old = *shared_var;
        *shared_var = old + 1;
    }
}

static void print_read_results(int *h_arr, int *d_arr, int N,
                               const char *label)
{
    int i;
    int maxNumToPrint = 10;
    int nToPrint = N > maxNumToPrint ? maxNumToPrint : N;
    hipMemcpy(h_arr, d_arr, nToPrint * sizeof(int),
                     hipMemcpyDeviceToHost);
    printf("Threads performing %s operations read values", label);

    for (i = 0; i < nToPrint; i++)
    {
        printf(" %d", h_arr[i]);
    }

    printf("\n");
}

int main(int argc, char **argv)
{
		
        int N = 64;
   	 	int block = 32;
    	int runs = 30;
    	int iters = 100000;
    	int r;
    	int *d_shared_var;
    	int h_shared_var_atomic, h_shared_var_unsafe;
    	int *d_values_read_atomic;
    	int *d_values_read_unsafe;
    	int *h_values_read;
        
       	

       	hipMalloc((void **)&d_shared_var, sizeof(int));
        hipMalloc((void **)&d_values_read_atomic, N * sizeof(int));
        hipMalloc((void **)&d_values_read_unsafe, N * sizeof(int));
        h_values_read = (int *)malloc(N * sizeof(int));

      	double atomic_mean_time = 0;
    	double unsafe_mean_time = 0;
       	
        my_timer timer1;
        my_timer timer2;
        for (r = 0; r < runs; r++)
    	{
        	timer1.start();
        	hipMemset(d_shared_var, 0x00, sizeof(int));
            hipLaunchKernelGGL(atomics,N / block, block,0,0,d_shared_var, d_values_read_atomic, N,
                                          iters);
  
        	hipDeviceSynchronize();
            timer1.stop();
        	atomic_mean_time += timer1.time_use/1000000;
        	hipMemcpy(&h_shared_var_atomic, d_shared_var, sizeof(int),hipMemcpyDeviceToHost);

        	timer2.start();
        	hipMemset(d_shared_var, 0x00, sizeof(int));
            hipLaunchKernelGGL(unsafe,N / block, block,0,0,d_shared_var, d_values_read_unsafe, N,iters);
        	hipDeviceSynchronize();
            timer2.stop();
        	unsafe_mean_time += timer2.time_use/1000000;
        	hipMemcpy(&h_shared_var_unsafe, d_shared_var, sizeof(int),hipMemcpyDeviceToHost);
    	}
        
        
        printf("In total, %d runs using atomic operations took %f s\n",
           runs, atomic_mean_time);
    	printf("  Using atomic operations also produced an output of %d\n",
           h_shared_var_atomic);
    	printf("In total, %d runs using unsafe operations took %f s\n",
           runs, unsafe_mean_time);
    	printf("  Using unsafe operations also produced an output of %d\n",
           h_shared_var_unsafe);

    	print_read_results(h_values_read, d_values_read_atomic, N, "atomic");
    	print_read_results(h_values_read, d_values_read_unsafe, N, "unsafe");

    	return 0;
}
