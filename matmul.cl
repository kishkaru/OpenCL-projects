__kernel void matmul(__global float *Y, __global float *A, __global float *B, int n)
{
	int idx = get_global_id(0); //gets the global work-item ID (threadID.x) of the thread.
	int idy = get_global_id(1); //gets the global work-item ID (threadID.y) of the thread.
	
	if((idx < n) && (idy < n)) 
		for(int i=0, j=0; i<n, j<n; i++, j++)
			Y[idx*n+idy] += A[idx*n+i] * B[j*n+idy];
}
