#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cmath>
#include <unistd.h>

#include "clhelp.h"

int main(int argc, char *argv[])
{
  std::string reduce_kernel_str;
  
  std::string reduce_name_str = 
    std::string("reduce");
  std::string reduce_kernel_file = 
    std::string("reduce.cl");

  cl_vars_t cv; 
  cl_kernel reduce;
  
  readFile(reduce_kernel_file,
	   reduce_kernel_str);
  
  initialize_ocl(cv);
  
  compile_ocl_program(reduce, cv, reduce_kernel_str.c_str(),
		      reduce_name_str.c_str());

  int *h_A, *h_Y;
  cl_mem g_Out, g_In;
  int n = (1<<24);

  int c;
  /* how long do you want your arrays? */
  while((c = getopt(argc, argv, "n:"))!=-1)
    {
      switch(c)
	{
	case 'n':
	  n = atoi(optarg);
	  break;
	}
    }
  if(n==0)
    return 0;

  h_A = new int[n];
  h_Y = new int[n];

  for(int i = 0; i < n; i++)
    {
      h_A[i] = 1;
      h_Y[i] = 0;
    }

  cl_int err = CL_SUCCESS;
  g_Out = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
			 sizeof(int)*n,NULL,&err);
  CHK_ERR(err);  
  g_In = clCreateBuffer(cv.context,CL_MEM_READ_WRITE,
			sizeof(int)*n,NULL,&err);
  CHK_ERR(err);

  //copy data from host CPU to GPU
  err = clEnqueueWriteBuffer(cv.commands, g_Out, true, 0, sizeof(int)*n,
			     h_Y, 0, NULL, NULL);
  CHK_ERR(err);


  err = clEnqueueWriteBuffer(cv.commands, g_In, true, 0, sizeof(int)*n,
			     h_A, 0, NULL, NULL);
  CHK_ERR(err);

/////////////////16.7M -> 32K///////////////
  size_t global_work_size[1] = {n};
  size_t local_work_size[1] = {512};

  double t0 = timestamp();

 /* Set Kernel Arguments */
  err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &g_In);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &g_Out);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 2, sizeof(cl_int)* local_work_size[0], NULL);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 3, sizeof(int), &n);
  CHK_ERR(err);
  
  /* Call kernel on the GPU */
  err = clEnqueueNDRangeKernel(cv.commands,
			       reduce,
			       1,//work_dim,
			       NULL, //global_work_offset
			       global_work_size, //global_work_size
			       local_work_size, //local_work_size
			       0, //num_events_in_wait_list
			       NULL, //event_wait_list
			       NULL //
			       );
  CHK_ERR(err);
  

  /////////////////32K -> 64///////////////
  global_work_size[1] = {n/512};
  local_work_size[1] = {512};
  int m = n/512; 
   
 /* Set Kernel Arguments */
  err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &g_Out);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &g_Out);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 2, sizeof(cl_int)* local_work_size[0], NULL);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 3, sizeof(int), &m);
  CHK_ERR(err);
  
  /* Call kernel on the GPU */
  err = clEnqueueNDRangeKernel(cv.commands,
			       reduce,
			       1,//work_dim,
			       NULL, //global_work_offset
			       global_work_size, //global_work_size
			       local_work_size, //local_work_size
			       0, //num_events_in_wait_list
			       NULL, //event_wait_list
			       NULL //
			       );
  CHK_ERR(err);

  /////////////////64 -> 1///////////////
  global_work_size[1] = {n/512/512};
  local_work_size[1] = {512};
  m = m/512; 
   
 /* Set Kernel Arguments */
  err = clSetKernelArg(reduce, 0, sizeof(cl_mem), &g_Out);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 1, sizeof(cl_mem), &g_Out);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 2, sizeof(cl_int)* local_work_size[0], NULL);
  CHK_ERR(err);
  err = clSetKernelArg(reduce, 3, sizeof(int), &m);
  CHK_ERR(err);
  
  /* Call kernel on the GPU */
  err = clEnqueueNDRangeKernel(cv.commands,
			       reduce,
			       1,//work_dim,
			       NULL, //global_work_offset
			       global_work_size, //global_work_size
			       local_work_size, //local_work_size
			       0, //num_events_in_wait_list
			       NULL, //event_wait_list
			       NULL //
			       );
  CHK_ERR(err);
  
  /////////////END//////////////////
  t0 = timestamp()-t0;
  
  //read result of GPU on host CPU
  err = clEnqueueReadBuffer(cv.commands, g_Out, true, 0, sizeof(int)*n,
			    h_Y, 0, NULL, NULL);
  CHK_ERR(err);
  
  int sum=0.0f;
  for(int i = 0; i < n; i++)
    {
      sum += h_A[i];
    }

  if(sum!=h_Y[0])
    {
      printf("WRONG: CPU sum = %d, GPU sum = %d\n", sum, h_Y[0]);
      printf("WRONG: difference = %d\n", sum-h_Y[0]);
    }
  else
    {
      printf("CORRECT: %d,%g\n",n,t0);
    }
 
  uninitialize_ocl(cv);
  
  delete [] h_A; 
  delete [] h_Y;
  
  clReleaseMemObject(g_Out); 
  clReleaseMemObject(g_In);
  
  return 0;
}
