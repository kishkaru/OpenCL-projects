__kernel void reduce(__global int *in, __global int *out, __local int *buf, int n) 
{
	size_t idx = get_global_id(0);
	size_t tid = get_local_id(0); 
	size_t gid = get_group_id(0);
	size_t dim = get_local_size(0);
	
	buf[tid] = in[idx]; //copy every element from input to local memory.
	barrier(CLK_LOCAL_MEM_FENCE);
	
	//for each element, do a reduce, O(logn) runtime.
	for(int offset = dim / 2; offset > 0; offset = offset / 2)
	{
		if (tid < offset) 
		{
			int ele = buf[tid];
			int other_ele = buf[tid + offset];

			buf[tid] = ele + other_ele;
		}
		
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	if (tid == 0) 
		out[gid] = buf[0]; //output the final reduced value.
}