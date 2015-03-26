////////////////////////////////////////////////////////////////////////////
///////////////////////////  Head file  :  C   ///////////////////////////////
////////////////////////////////////////////////////////////////////////////


// OpenGL Graphics includes
//#include "arrayfire.h"
//using namespace af;

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>

extern "C"
{
	
	#include <GL/glew.h>
	#include <GL/freeglut.h>
};

#include "cuda_gl_interop.h"
#include <vector_types.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"
#include "texture_fetch_functions.h"


#include "cublas_v2.h"
#include "cufft.h"

#include <complex>
using namespace std;

//#pragma comment (lib,"glut64")
#pragma comment (lib,"cublas")
#pragma comment (lib,"cufft")
#pragma comment (lib,"glew64")
#pragma comment (lib,"freeglut")
//#pragma comment (lib,"libafcu")

#define PI 3.1415926535897931e+0
#define EPS 2.2204e-16
#define RANK 2
#define IMGNUM 6
#define IMG_W 1280
#define IMG_H 1024
#define imageH2 1024
#define IDX2R(x,y,N2) (((x)*(N2))+(y))
#define IDX2C(x,y,N1) (((y)*(N1))+(x))

texture<double, 1, cudaReadModeElementType>tex_wx;
texture<double, 1, cudaReadModeElementType>tex_wy;

__constant__  int dev_WH[3] = {1280, 1024, 1280*1024};
__constant__ double dev_Light[18] =
{
	-0.002509321144440, -0.951463164864384,-0.911878217778859,-0.021329590025452,0.952225888653372,0.913936545603041,
	1.081291324462186,0.610320422352033,-0.537682637919354,-1.120862939242518,-0.608106656857313,0.571590722912323,
	-0.176751166356379,-0.153494830074108,-0.153528584102128,-0.173923275907694,-0.195869030923602,-0.195684490937402
};

__constant__ cuDoubleComplex J = {0, -1};
__constant__ cuDoubleComplex complexEPS = {EPS,0};

//loading picture functions
extern "C" void LoadBMPFile(unsigned char **dst, int *offset, int *width, int *height, const char *name);
extern "C" void LoadManyBmp(unsigned char **dst, int offset, int width, int height,  char **nameList);

//count Z
__global__ void kernel_count_pq(cuDoubleComplex *dev_p,cuDoubleComplex*dev_q,uchar1 *dev_Mask,uchar1 *dev_Src, int width, int height);
__global__ void kernel_count_Wxy(double *wx, double *wy , int width, int height);
__global__ void kernel_count_Cw(double *,double *,cuDoubleComplex*, cuDoubleComplex*, cuDoubleComplex* , int width, int height);

__global__ void kernel_copy_z2gl(float4 *gl_Z, double *Z, int w, int h,float g_fAnim);
__global__ void kernel_tmp_minmax(cuDoubleComplex*dev_p ,double *minmax , int width ,int height);
