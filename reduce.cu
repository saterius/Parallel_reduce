#include<stdio.h>
#include<math.h>

#define N 2048

//Interleave addressing kernel version
__global__ void interleaved_reduce(int* d_in, int* d_out) {

	int i = threadIdx.x;
	int M = N/2;
	__shared__ int sharedMem[N];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	sharedMem[i] = d_in[id];
	__syncthreads();
	for(int s = 1; s < N; s = s<<1) {
		if(i < M) {
			//printf("stride = %d, thread %d is active \n", s, i);
			sharedMem[(2*s)*id] = sharedMem[(2*s)*id] + sharedMem[(2*s)*id+s];
		}
		__syncthreads();
		M = M/2;
	}
	if(i == 0)
		d_out[0] = sharedMem[0];
}

//Contiguous addressing kernel version
__global__ void contiguous_reduce(int* d_in, int* d_out){

	int i = threadIdx.x;
	int M = N/2;
	__shared__ int sharedMem[N];
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	sharedMem[i] = d_in[id];
	__syncthreads();
	for(int s = M; s > 0; s = s>>1) {
		if(i < M) {
			//printf("stride = %d, thread %d is active \n", s, i);
			sharedMem[id] = sharedMem[id] + sharedMem[id+s];
		}
		__syncthreads();
		M = M/2;
	}
	if(i == 0)
		d_out[0] = sharedMem[0];
}

int main()
{
	int h_in[N];
	int h_out = 0;

	for(int i = 0; i < N; i++)
		h_in[i] = i+1;

	int *d_in, *d_out;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**) &d_in, N*sizeof(int));
	cudaMalloc((void**) &d_out, sizeof(int));
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);

	cudaEventRecord(start);

	//kernel call
	//interleaved_reduce<<<1, 1024>>>(d_in, d_out);
	contiguous_reduce<<<1, 1024>>>(d_in, d_out);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);

	//printf("Output: %d\n", h_out);
	printf("%f\n", milliseconds);
	
	return -1;
}
