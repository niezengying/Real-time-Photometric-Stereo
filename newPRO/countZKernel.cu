////////////////////////////////////////////////////////////////////////////
//////////////////////   Calculate Z functions  :  CUDA C   /////////////////////////
////////////////////////////////////////////////////////////////////////////

#include "newPro.h"

#include <helper_functions.h> 
#include "cuda_runtime.h"
#include <helper_cuda.h>
#include "device_launch_parameters.h"
#include <timer.h>

// Reconstruction data
extern uchar1 *dev_Mask, *dev_Src;  
extern double *dev_wx, *dev_wy,*dev_Z;
extern cuDoubleComplex *dev_Zx, *dev_Zy, *dev_Cw ;
extern	float4* dev_gl_Z ;

extern 	cufftHandle pland2zP, pland2zQ, planz2d_inv;
extern cublasHandle_t handle_Z_min,handle_Z_max;
extern int minZ_idx,maxZ_idx;

// BMP data
extern int imageW, imageH, bmpHdrOff;
extern unsigned char *h_Src,*h_Mask;
extern char *imagePath[];
extern char *maskPath;

// Run time
cudaEvent_t start, stop;
float elapsedTime;

////设备内存申请与释放
/////////////////////Malloc && Free//////////////////////////////
extern "C" void CudaMalloc()
{
	checkCudaErrors(cudaSetDevice(0));
	printf("CUDA Malloc...\n");
	
	//malloc Mask
	checkCudaErrors(cudaMalloc((void**)&dev_Mask,imageW*imageH*sizeof(uchar1)));
	checkCudaErrors(cudaMemset(dev_Mask,0,imageH*imageW*sizeof(uchar1)));

	//malloc Src
	checkCudaErrors(cudaMalloc((void**)&dev_Src,imageW*imageH*sizeof(uchar1)*IMGNUM));
	checkCudaErrors(cudaMemset(dev_Src,0,imageH*imageW*sizeof(uchar1)*IMGNUM));

	//malloc wx && wy
	checkCudaErrors(cudaMalloc((void**)&dev_wx, imageH*imageW*sizeof(double)));
	checkCudaErrors(cudaMalloc((void**)&dev_wy, imageH*imageW*sizeof(double)));
	//cudaChannelFormatDesc desc_wxy = cudaCreateChannelDesc<double>();
	//cudaBindTexture(NULL,tex_wx,dev_wx,desc_wxy,imageH*imageW*sizeof(double));
	//cudaBindTexture(NULL,tex_wy,dev_wy,desc_wxy,imageH*imageW*sizeof(double));
	checkCudaErrors(cudaMemset(dev_wx,0,imageH*imageW*sizeof(double)));
	checkCudaErrors(cudaMemset(dev_wy,0,imageH*imageW*sizeof(double)));

	//malloc dev_fourier_pq 
	cudaMalloc((void**)&dev_Zx,sizeof(cuDoubleComplex)*imageW*(imageH));
	cudaMalloc((void**)&dev_Zy,sizeof(cuDoubleComplex)*imageW*(imageH));
	checkCudaErrors(cudaMemset(dev_Zx,0,sizeof(cuDoubleComplex)*imageW*(imageH)));
	checkCudaErrors(cudaMemset(dev_Zy,0,sizeof(cuDoubleComplex)*imageW*(imageH)));

	//malloc dev_Cw
	checkCudaErrors(cudaMalloc((void**)&dev_Cw, (imageH)*imageW*sizeof(	cuDoubleComplex)));
	checkCudaErrors(cudaMemset(dev_Cw,0, (imageH)*imageW*sizeof(cuDoubleComplex)));
	
	//malloc dev_Z
	checkCudaErrors(cudaMalloc((void**)&dev_Z, imageH*imageW*sizeof(double)));
	checkCudaErrors(cudaMemset(dev_Z,0, imageH*imageW*sizeof(double)));

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	LoadBMPFile(&h_Mask, &bmpHdrOff, &imageW, &imageH, maskPath);
	LoadManyBmp(&h_Src,bmpHdrOff,imageW, imageH,imagePath);
	printf("%d, %d\n",imageW,imageH);

}

extern "C" void CudaFree()
{
	cudaFree(dev_Mask);
	cudaFree(dev_Src);
	cudaFree(dev_wx);
	cudaFree(dev_wy);
	cudaFree(dev_Zx);
	cudaFree(dev_Zy);
	cudaFree(dev_Cw);
	cudaFree(dev_Z);
}

extern "C" void load_Image()
{
	LoadBMPFile(&h_Mask, &bmpHdrOff, &imageW, &imageH, maskPath);
	LoadManyBmp(&h_Src,bmpHdrOff,imageW, imageH,imagePath);
	checkCudaErrors(cudaMemcpy(dev_Src,h_Src,imageW*imageH*sizeof(uchar1)*IMGNUM,cudaMemcpyHostToDevice));
	free(h_Mask);
	free(h_Src);
}

///////////////////////Calculate the median Value//////////////////////
//count Wx && Wy
extern "C" void launch_count_Wxy(double *dev_wx,double *dev_wy, int width, int height)
{
	dim3 block_wxy(8,8,1);   //每个块的线程数
	dim3 grid_wxy((width+block_wxy.x-1)/block_wxy.x, (height+block_wxy.y-1)/block_wxy.y,1); //每个格的块数
	kernel_count_Wxy<<<grid_wxy,  block_wxy>>>(dev_wx,dev_wy, width, height);
}

//count Zx && Zy
extern "C" void launch_count_PQ(cuDoubleComplex *dev_Zx,cuDoubleComplex *dev_Zy,int width, int height)
{
	dim3	dimBlock(8,8,1);
    dim3	dimGrid((width+dimBlock.x-1)/dimBlock.x, (height+dimBlock.y-1)/dimBlock.y,1);
	kernel_count_pq<<<dimGrid,dimBlock>>>(dev_Zx,dev_Zy,dev_Mask,dev_Src,width,height);
}

//count C(w)
extern "C" void launch_count_Cw(cuDoubleComplex *dev_Cw, int width, int height)
{
	dim3 block_Cw(8,8,1);
	dim3 grid_Cw((width+block_Cw.x-1)/block_Cw.x, (height+block_Cw.y-1)/block_Cw.y,1);
	kernel_count_Cw<<<grid_Cw,  block_Cw>>>(dev_wx,dev_wy,dev_Zx,dev_Zy,dev_Cw, width, height);
}

//count the result Z
extern "C" void launch_count_z2gl(float4 *dev_gl_Z,int width, int height,float time)
{	
	dim3 block_z2gl(8,8,1);
	dim3 grid_z2gl((width+block_z2gl.x-1)/block_z2gl.x, (height+block_z2gl.y-1)/block_z2gl.y,1);
	kernel_copy_z2gl<<<grid_z2gl, block_z2gl>>>(dev_gl_Z,dev_Z, width, height,time);
}

////测试代码不用关注
///////////////////////////////Test Code//////////////////////////////
extern "C" void test_cplx_SumMinMax(cuDoubleComplex *src,int width, int height)
{	
	double2 p,q;
	double sumP;
	checkCudaErrors(cublasDzasum(handle_Z_min, width*height, src, 1, &sumP));
	checkCudaErrors(cublasIzamin(handle_Z_min, width*height, src, 1, &minZ_idx));
	checkCudaErrors(cublasIzamax(handle_Z_max,width*height, src, 1, &maxZ_idx));

	checkCudaErrors(cudaMemcpy(&p,&src[minZ_idx-1],sizeof(double2),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&q,&src[maxZ_idx-1],sizeof(double2),cudaMemcpyDeviceToHost));

	//printf("Result:\t%f\t%f\t%f\n",sumP,abs(p.x)+abs(p.y),abs(q.x)+abs(q.y));
	printf("Result:\t%f\t%f\t%f\n",sumP,p.x,q.x);
}

extern "C" void test_dbl_SumMinMax(double *src,int width, int height)
{
	double p,q;
	double sumP;
	checkCudaErrors(cublasDasum(handle_Z_min, width*height, src, 1, &sumP));
	checkCudaErrors(cublasIdamin(handle_Z_min, width*height, src, 1, &minZ_idx));
	checkCudaErrors(cublasIdamax(handle_Z_max,width*height, src, 1, &maxZ_idx));

	checkCudaErrors(cudaMemcpy(&p,&src[minZ_idx-1],sizeof(double),cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(&q,&src[maxZ_idx-1],sizeof(double),cudaMemcpyDeviceToHost));

	printf("Result:\t%f\t%f\t%f\n",sumP,p,q);
}

template <class T, class T2>
 void test_rand_Print(T *src,T2 *src2,int width, int height)
{
	double *h_p, *h_p2;
	h_p = (double *)malloc(width*height*sizeof(T));
	h_p2 = (double *)malloc(width*height*sizeof(T2));
	cudaMemcpy(h_p,src,width*height*sizeof(T),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_p2,src2,width*height*sizeof(T2),cudaMemcpyDeviceToHost);

	int ii = 0,jj = 0;
	int N1 = sizeof(T)/sizeof(double);
	int N2 = sizeof(T2)/sizeof(double);
	while((scanf("%d %d",&ii,&jj)) == 2 && ii != EOF)
	{
			ii = ii - 1 ;
			jj = jj - 1  ;
			if(N1>1)
				printf("%f\t%f\n",h_p[IDX2C(ii,jj,width) * N1], h_p[IDX2C(ii,jj,width)  * N1 + 1]);
			else
				printf("%f\t%f\n",h_p[IDX2C(ii,jj,width)]);

			if(N2>1)
				printf("%f\t%f\n", h_p2[IDX2C(ii,jj,width) * N2], h_p2[IDX2C(ii,jj,width)  * N2 + 1]);
			else
				printf("%f\t%f\n", h_p2[IDX2C(ii,jj,width)]);
	}

	free(h_p);
	free(h_p2);
}

 extern "C" void test_Print2File(double * src, int width, int height)
{
	double *h_p;		
	FILE *stream;
	h_p = (double *)malloc(width*height*sizeof(double));
	checkCudaErrors(cudaMemcpy(h_p,src,width*height*sizeof(double),cudaMemcpyDeviceToHost));
	if((stream = fopen("text.txt", "w+"))== NULL)
		printf("error!\n");

	printf("%d,%d\n",width,height);
	for(int j = 0; j < height ;j++)
		for(int i = 0; i < width; i++)
		{
			fprintf(stream,"%f\t%f\t%f\n",(float)i,(float)j,(float)(h_p[i+width*j])*(-0.5));
		}
	fclose(stream); 
	free(h_p);
}

 ///代码调用主体
/////////////////////////Calculate Function//////////////////////
extern "C" void testCuda(struct cudaGraphicsResource **cuda_vbo_resource,int width, int height,float g_fAnim)
{
	cudaEventRecord(start,0);
	size_t size;

	//load_Image();
	checkCudaErrors(cudaMemcpy(dev_Mask,h_Mask,width*height*sizeof(uchar1),cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_Src,h_Src,width*height*sizeof(uchar1)*IMGNUM,cudaMemcpyHostToDevice));

	printf("\n0.	count  Wx && Wy !!! \n");
	printf("\n1.	count Zx && Zy !!! \n");
	launch_count_PQ(dev_Zx,dev_Zy,width, height);
	
	printf("\n2.	dev_pq FFT == Cx !!! \n");
	checkCudaErrors(cufftExecZ2Z(pland2zP, dev_Zx, dev_Zx,CUFFT_FORWARD));
	checkCudaErrors(cufftExecZ2Z(pland2zQ, dev_Zy, dev_Zy,CUFFT_FORWARD));

	printf("\n2.5.	Wxy  iFFTshift  !!! \n"); 
	printf("\n3.	count C(w) == Wx * Cx !!!\n");	//  Frankt-Chellappa Algrotihm
	launch_count_Cw(dev_Cw, width, height);	// Minimize Cw in this way
	//test_dbl_SumMinMax(dev_Cw,width, height);

	printf("\n4.	Z Reconstruction == C(w) IFFT !!!\n");
	checkCudaErrors(cufftExecZ2Z(planz2d_inv, dev_Cw, dev_Cw,CUFFT_INVERSE));	// Reconstruction 
	cudaMemcpy2D(dev_Z,sizeof(double),dev_Cw,sizeof(double2),sizeof(double),width*height,cudaMemcpyDeviceToDevice);	//Get the real part of Z
	//test_rand_Print<double,double2>(dev_Z, dev_Cw, width, height);

	//checkCudaErrors(cublasIdamax(handle_Z_max,width*height, dev_Z, 1, &maxZ_idx));

	printf("\n5.	bind	Z to opengl !!!\n");    ////将计算结果绑定OpenGL
	checkCudaErrors(cudaGraphicsMapResources(1,cuda_vbo_resource,NULL));
	checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&dev_gl_Z,&size,*cuda_vbo_resource));
	launch_count_z2gl(dev_gl_Z,width, height,g_fAnim);
	checkCudaErrors(cudaGraphicsUnmapResources(1,cuda_vbo_resource,0)); 

	//Show the time
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime,start,stop);
	printf("\ntime to generate:\t %3.1f ms\n\n",elapsedTime);
}