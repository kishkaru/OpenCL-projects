__kernel void vvadd (__global float *Y, __global float *A, __global float *B, int n)
{
	int idx = get_global_id(0); //gets the global work-item ID (threadID) of the thread.
	
	if(idx < n) {
		Y[idx] = A[idx] + B[idx]; //grabs elements from each array with this work-item ID, add together, and store into Y.
	}	
	
}
