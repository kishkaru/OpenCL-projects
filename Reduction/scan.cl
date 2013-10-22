//this kernel is needed to generate the final "large" triangle. It takes each element and adds the largest element of the previous
//triangle to "grow" the triangle.
__kernel void update(__global int *in, __global int *block, int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  if(idx < n && gid > 0)
    {
      in[idx] = in[idx] + block[gid-1]; //adds each element to the largest element of the previous triangle
    }
}

//this kernel does the meat of the work, generating the "sawtooth"-like triangles for each call to scan()
__kernel void scan(__global int *in, __global int *out, __global int *bout, __local int *buf, int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

	buf[tid] = in[idx];  //copy every element from input to local memory
	barrier(CLK_LOCAL_MEM_FENCE); //make sure all elements are copied into local mem.
	
	out[idx] = 0; //zero out the output array (to remove any dependencies with previous recursive runs)
	
	for(int i=0; i<=tid; i++)
	{
		out[idx] += buf[i];  //output the partially scanned data
	}
	
	if(tid == dim-1)
	{
		bout[gid] = out[idx]; //output the largest of each scanned data to be used for final scan by update_kern
	}
  
}



