////////////////////////////////////////////////////////////////////////////
////////  Kernel Declarations and Definitions  :  CUDA C  ///////////////
////////////////////////////////////////////////////////////////////////////

#include "newPro.h"
#include <helper_functions.h> 
#include <helper_cuda.h>

//count wx && wy
__global__  void kernel_count_Wxy(double *wx, double *wy , int width, int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;           //通过块id和线程id，求当前线程在所有线程中的索引                                                                                                                                                                               
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = x + y * width;

	//int offset2 = x + (height - y -1) * width;
	if(  x < width && y < height)
	{
		wx[offset] = PI * ((x/(width-1.0)) - 0.5 );
		//wy[offset] = PI * ((y/(height-1.0)) - 0.5 );
		//wx[offset] = PI * (0.5 - (x/(width-1.0)));
		wy[offset] = PI * ( 0.5 - (y/(height-1.0)));
	}
}

//count zx && zy 
__global__ void kernel_count_pq(cuDoubleComplex *dev_p,cuDoubleComplex *dev_q, uchar1 *dev_Mask, uchar1 *dev_Src,int width, int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	//int offset = x + y * blockDim.x * gridDim.x;  
	int offset = x + y * width;
	double tmp_p,tmp_q;
	double N[3] = {0,0,0};
	double img[6] = {0,0,0,0,0,0};
	int count = width * height;

	if(dev_Mask[offset].x >25 && x<width && y<height)
	{
		for(int i = 0; i<6;i++)
		{
			img[i] = dev_Src[offset+i *count].x;
			N[0] += dev_Light[i]*img[i];
			N[1] += dev_Light[i+6]*img[i];
			N[2] += dev_Light[i+12]*img[i];
		}

		tmp_p = (-1.0)*N[0]/(N[2]+EPS);
		tmp_q = (-1.0)*N[1]/(N[2]+EPS);
		dev_p[offset ] = make_cuDoubleComplex(tmp_p,0);
		dev_q[offset ] = make_cuDoubleComplex(tmp_q,0);
	}
	else
	{
		dev_p[offset ] = make_cuDoubleComplex(0,0);
		dev_q[offset ] = make_cuDoubleComplex(0,0);
	}	
}

//count Cw
__global__ void kernel_count_Cw(double *wx, double *wy, cuDoubleComplex *DZDX, cuDoubleComplex *DZDY, cuDoubleComplex *tmp_z, int width, int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	int i = IDX2C(x,y,width);
	int N = width*height;

	int xx = ( x + width/2 ) % width;
	int yy = ( y + height/2) % height;
	int j = IDX2C(xx,yy,width);  //fftshift
	
	if (x < width && y < height )
	{
		double dd = wx[j] *wx[j] + wy[j] * wy[j] ;

		cuDoubleComplex tmp;
		tmp.x = ((wx[j]*DZDX[i].x)+(wy[j]*DZDY[i].x))/(dd*N +EPS);
		tmp.y = ((wx[j]*DZDX[i].y)+(wy[j]*DZDY[i].y))/(dd*N +EPS);
		tmp_z[i] = cuCmul(J,tmp);
	}
	else
	{
		tmp_z[i].x = 0;
		tmp_z[i].y = 0;
	}
}

//calculate coordinates
__global__ void kernel_copy_z2gl(float4 *gl_z, double *Z, int width, int height,  float time)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
	int offset = y*width+x;

	// calculate uv coordinates
	//float u =x /(float) (width-1);
	// float v =y /(float) (height-1);
	float u =x /(float) width;              //将坐标归一化到0,1之间
   float v =y /(float) width;
	u = u*2.0f-1.0f;
	v = v*2.0f-1.0f;

	// write output vertex
	float w = (float)Z[offset]/width;
    gl_z[offset] = make_float4(u, w, v, 1.0f);
}


__global__ void kernel_tmp_minmax(cuDoubleComplex *data ,double *minmax , int width ,int height)
{
	unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

	int offset = y*width +x;
	
	__shared__ double min_p ;
	__shared__ double max_p;
	
	if (threadIdx.x == 0 && threadIdx.y == 0) {
       min_p = 0;
	   max_p = 0;
	}
	__syncthreads();

	if(data[offset].x < min_p)
	{
		min_p = data[offset].x;
		__syncthreads();
	}
	else if(data[offset].x > max_p)
	{
		max_p = data[offset].x;
		__syncthreads();
	}
	
	if (threadIdx.x == 0 && threadIdx.y == 0) {
		minmax[0] = min_p;
		minmax[1] = max_p;
	}
	__syncthreads();
}