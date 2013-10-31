__kernel void update(__global int *in,
		     __global int *block,
		     int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);

  if(idx < n && gid > 0)
    {
      in[idx] = in[idx] + block[gid-1];
    }
}

__kernel void reassemble(__global int *in, 
		   __global int *out, 
		   __global int *g_zeros,
		   __global int *g_ones,
		   int k,
		   int n)
{
	size_t idx = get_global_id(0);
	
	__private int temp_i;
	__private int out_idx;
	
	if(idx < n){
		temp_i = (in[idx] >> k) & 0x1;
	
		if(temp_i)
			out_idx = g_zeros[n-1] + g_ones[idx] - 1;
		else
			out_idx = g_zeros[idx] - 1;
		
		out[out_idx] = in[idx];
	}
		
}		   
		   

__kernel void scan(__global int *in, 
		   __global int *out, 
		   __global int *bout,
		   /* dynamically sized local (private) memory */
		   __local int *buf, 
		   int v,
		   int k,
		   int n)
{
  size_t idx = get_global_id(0);
  size_t tid = get_local_id(0);
  size_t dim = get_local_size(0);
  size_t gid = get_group_id(0);
  int t, r = 0, w = dim;

  //reads elements from the input into the local buffer
  if(idx<n)
    {
      t = in[idx];
      t = (v==-1) ? t : (v==((t>>k)&0x1)); 
      buf[tid] = t;
    }
  else
    {
      buf[tid] = 0;
    }
	
	//creates a "second" buffer from the 2nd half of buf, for temporarily
	//storing the outputs. This buffer will be used as the input for the
	//next iteration of the loop. This switching will happen at every 
	//iteration. "buf" will always point to the input/read and buf2 will
	//always point to the output/write.
	

	//run the algorithm for array size from dim to 1, halfing the number of
	//threads used each active because each thread grabs itself and its 
	//assigned "neighbor" depending on "d", which is the division size.
	for(int d = 1; d < dim; d = d*2)
	{
		if((tid >= d) && (idx < n))
		{
			buf2[tid] = buf[tid] + buf[tid - d];
		}
		else
		{
			buf2[tid] = buf[tid];
		}
				
		barrier(CLK_LOCAL_MEM_FENCE);
		
		__local int *tmp = buf; //switch the two buffer pointers
		buf = buf2;
		buf2 = tmp;
	
		barrier(CLK_LOCAL_MEM_FENCE);
	}
  
  //stores partial scans in the output array
  if(idx < n)
    {
      out[idx] = buf[r+tid];
    }
  //stores the work group's total partial "reduction" in the array bout, to be used by further recursive calls to scan
  if(tid==0)
    {
      bout[gid] = out[r+dim-1];
    }
}



