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
	
	
	//__local int *buf2 = buf + dim;
	
	barrier(CLK_LOCAL_MEM_FENCE);

  //iterates through, reducing by half each time and summing scanned elements
	for(int d = 1; d < dim; d = d*2)
	{
		if(tid >= d)
		{
			buf[w+tid] = buf[r+tid] + buf[r+tid - d];
		}
		else
		{
			buf[w+tid] = buf[r+tid];
		}
			
		barrier(CLK_LOCAL_MEM_FENCE);
		
		if(tid == 0)
		{
			int tmp = r;
			r = w;
			w = tmp;
		}
		
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



