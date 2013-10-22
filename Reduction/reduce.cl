__kernel void reduce(__global int *in, __global int *out, 
		     __local int *buf, int n)
{
  size_t tid = get_local_id(0);
  size_t gid = get_group_id(0);
  size_t dim = get_local_size(0);
  size_t idx = get_global_id(0);
}
