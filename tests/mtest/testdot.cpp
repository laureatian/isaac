#include <cmath>
#include <iostream>
#include <type_traits>
#include <CL/cl.h>
#include <unistd.h>
#include "api.hpp"
#include "isaac/array.h"
#include "isaac/driver/common.h"
#include "clBLAS.h"
#include "cublas.h"
#include "half.hpp"



//using myhalf=isaac::half;
using myhalf =  half_float::half;
void test_half(){
int N = 10;
//float alpha = 2.2f;
int inc_a=1;
int off_a =0;
int inc_b=1;
int off_b=0;
cl_int err;
cl_uint numPlatforms;
cl_platform_id firstPlatformId;
cl_context context=NULL;
err = clGetPlatformIDs(1,&firstPlatformId,&numPlatforms);
cl_context_properties contextProperties[]={CL_CONTEXT_PLATFORM,(cl_context_properties)firstPlatformId,0};
context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_GPU,NULL,NULL,&err);
cl_device_id device;
cl_device_id *devices;
cl_command_queue commandeQueue = NULL;
size_t deviceBufferSize = -1;
err = clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceBufferSize);
devices =  new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
err = clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceBufferSize,devices,NULL);
commandeQueue = clCreateCommandQueue(context,devices[0],0,NULL);
device = devices[0];
delete devices;
float  x=1.0f;
myhalf alpha=myhalf(x);
myhalf result[1];
myhalf a[10];
myhalf b[10];
size_t d[1];
d[0]=10;
for(int i=0;i<10;i++){
 float  ai=0.5*i;
 float  bi=2.0*i;
 std::cout<<"aibi"<<ai<<"  "<<bi<<std::endl;
  a[i]=myhalf(ai);
  b[i]=myhalf(bi);
}
for(int i=0;i<10;i++){
  std::cout<<"a[i] "<<a[i]<<std::endl;
}
for(int i=0;i<10;i++){
  std::cout<<"b[i] "<<b[i]<<std::endl;
}
cl_mem am =clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(myhalf)*10,a,NULL);
cl_mem bm =clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(myhalf)*10,b,NULL);
cl_mem cm = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(myhalf),NULL,NULL);
cl_mem dm =clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(myhalf)*12,NULL,NULL);
int status =clblasHdot(10,cm ,0, am, 0, 1, bm , 0, 1,dm, 1, &commandeQueue, 0, NULL, NULL);
std::cout<<"status"<<status<<std::endl;

err = clEnqueueReadBuffer(commandeQueue,cm,CL_TRUE,0,sizeof(myhalf),result,0,NULL,NULL);

for(int i=0;i<10;i++){
  std::cout<<"result[i] "<<result[i]<<std::endl;

}
clReleaseContext(context);
//clRealeaseKernel();
clReleaseMemObject(am);
clReleaseMemObject(bm);
clReleaseCommandQueue(commandeQueue);
clReleaseDevice(device);
}


void test_float(){
int N = 10;
//float alpha = 2.2f;
int inc_a=1;
int off_a =0;
int inc_b=1;
int off_b=0;
cl_int err;
cl_uint numPlatforms;
cl_platform_id firstPlatformId;
cl_context context=NULL;
err = clGetPlatformIDs(1,&firstPlatformId,&numPlatforms);
cl_context_properties contextProperties[]={CL_CONTEXT_PLATFORM,(cl_context_properties)firstPlatformId,0};
context = clCreateContextFromType(contextProperties,CL_DEVICE_TYPE_GPU,NULL,NULL,&err);
cl_device_id device;
cl_device_id *devices;
cl_command_queue commandeQueue = NULL;
size_t deviceBufferSize = -1;
err = clGetContextInfo(context, CL_CONTEXT_DEVICES,0,NULL,&deviceBufferSize);
devices =  new cl_device_id[deviceBufferSize / sizeof(cl_device_id)];
err = clGetContextInfo(context,CL_CONTEXT_DEVICES,deviceBufferSize,devices,NULL);
commandeQueue = clCreateCommandQueue(context,devices[0],0,NULL);

device = devices[0];
delete devices;
float x=2.0f;
float alpha=float(x);
float result[10];
float a[10];
float b[10];
size_t d[1];
d[0]=10;
for(int i=0;i<10;i++){
  a[i]=0.5*i;
  b[i]=2.0*i;
}
for(int i=0;i<10;i++){
  std::cout<<"a[i] "<<a[i]<<std::endl;
}
for(int i=0;i<10;i++){
  std::cout<<"b[i] "<<b[i]<<std::endl;
}
cl_mem am =clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(float)*10,a,NULL);
cl_mem bm =clCreateBuffer(context,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,sizeof(float)*10,b,NULL);
cl_mem cm = clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float),NULL,NULL);
cl_mem dm =clCreateBuffer(context,CL_MEM_READ_WRITE,sizeof(float)*10,NULL,NULL);
//clblasSaxpy(10, alpha, am, 0, 1, bm , 0, 1, 1, &commandeQueue, 0, NULL, NULL);
int status=3;
status =clblasSdot(10,cm ,0, am, 0, 1, bm , 0, 1,dm, 1, &commandeQueue, 0, NULL, NULL);
//status =clblasSaxpy(10, alpha, am, 0, 1, bm , 0, 1, 1, NULL, 0, NULL, NULL);
sleep(5);
std::cout<<"status"<<status<<std::endl;
err = clEnqueueReadBuffer(commandeQueue,cm,CL_TRUE,0,sizeof(float),result,0,NULL,NULL);
sleep(3);
for(int i=0;i<10;i++){
  std::cout<<"result[i] "<<result[i]<<std::endl;
}

clReleaseContext(context);
//clRealeaseKernel();
clReleaseMemObject(am);
clReleaseMemObject(bm);
clReleaseCommandQueue(commandeQueue);
clReleaseDevice(device);
}

int main(){
//  test_float();
  test_half();

std::cout<<"That is OK"<<std::endl;

}
