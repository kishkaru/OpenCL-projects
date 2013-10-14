// The kernel. 
// __global is an array in the global memory. N is the size of array.

__kernel void incr (__global float *Y, int n)
{
  int idx = get_global_id(0); //gets the global work-item ID (threadID) of the thread.
  if(idx < n)
    {
      Y[idx] = Y[idx] + 1.0f; //grab the element from the array with this work-item ID, add 1 and store back.
    }
}
