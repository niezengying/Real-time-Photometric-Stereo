////////////////////////////////////////////////////////////////////////////
//////////////////////   OpenGL functions :  C   ///////////////////////////
////////////////////////////////////////////////////////////////////////////

// OpenGL Graphics includes
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

#include <helper_functions.h> 
#include <helper_cuda.h>   
#include <helper_cuda_gl.h>

#include "cublas_v2.h"
#include "cufft.h"

// constants
int imageW =1280, imageH =1024, bmpHdrOff;
unsigned int windowW =640, windowH = 512;
const unsigned int meshW =1280, meshH =1024;
int NRANK[2] = {imageH, imageW};

// openGL vertex buffers
GLuint vbo;
struct cudaGraphicsResource *cuda_posVB_resource,  *cuda_vbo_resource;;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 20.0, rotate_y = 0.0;
float translate_x = 0.0f, translate_y = 0.0f, translate_z = -2.0f;

bool animate = true;
bool drawPoints = true;
float g_fAnim = 0.0f;
StopWatchInterface *timer = NULL;

// pointers to device object
unsigned char *h_Src,*h_Mask;
uchar1 *dev_Mask, *dev_Src;
double *dev_wx, *dev_wy,*dev_Z;
cuDoubleComplex *dev_Cw, *dev_Zx, *dev_Zy;
float4* dev_gl_Z ;

// handle
cufftHandle pland2zP, pland2zQ,planz2d_inv;
cublasHandle_t handle_Z_min,handle_Z_max;
int minZ_idx,maxZ_idx;

char *maskPath = "bmp/mask.bmp";
char *imagePath[] = {
		"bmp/0.bmp",
		"bmp/1.bmp",
		"bmp/2.bmp",
		"bmp/3.bmp",
		"bmp/4.bmp",
		"bmp/5.bmp"
};

extern "C" void testCuda(struct cudaGraphicsResource **cuda_vbo_resource, int width,int height, float testCuda);
extern "C" void launch_kernel(float4 * dev_gl_Z,int width, int height, float time);

int initGL(int *argc,char **argv);
bool runTest(int argc, char **argv);
static void display(void);
static void keyboard(unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);
void reshape(int w, int h);
void timerEvent(int value);
void createVBO(GLuint *vbo,int size);
void deleteVBO(GLuint *vbo);
void cleanup();
void createMeshIndexBuffer(GLuint *id, int w, int h);

extern "C" void CudaMalloc();
extern "C" void CudaFree();
extern "C" void LoadBMPFile(unsigned char  **dst, int *offset,int *width, int *height, const char *name);
extern "C" void LoadManyBmp( unsigned char **dst, int offset, int width, int height, char **nameList);
extern "C" void launch_count_Wxy(double *dev_wx,double *dev_wy, int width, int height);


//host code
int main(int argc,char **argv)
{
	cudaDeviceReset();
	runTest(argc, argv);
	cudaDeviceReset();
	system("pause");
	return 0;
}

//OpenGL入口，对数据进行初始化等
bool runTest(int argc, char **argv)
{
	cudaGLSetGLDevice(gpuGetMaxGflopsDeviceId());
	sdkCreateTimer(&timer);
	
	if(false == initGL(&argc, argv))  //init openGL
	{
		cudaDeviceReset();
		return false;
	}

	unsigned int size = imageW * imageH *sizeof(float4);
	glutCloseFunc(cleanup);

	CudaMalloc();
	createVBO(&vbo,imageW * imageH *sizeof(float4));

	//createMeshPositionVBO(&vbo, imageW,imageH);
	cufftPlan2d(&pland2zP,   imageH, imageW, CUFFT_Z2Z ); //create handles
	cufftPlan2d(&pland2zQ,  imageH, imageW,CUFFT_Z2Z );
	cufftPlan2d(&planz2d_inv, imageH,  imageW, CUFFT_Z2Z) ;

	cublasCreate(& handle_Z_min);
	cublasCreate(& handle_Z_max);
	launch_count_Wxy(dev_wx,dev_wy, imageW, imageH);

	glutDisplayFunc(display);  //显示函数
	glutKeyboardFunc(keyboard);
	glutMotionFunc(motion);  //鼠标操作
	glutMouseFunc(mouse);
	glutReshapeFunc(reshape);
	glutTimerFunc(0, timerEvent,0);

	glutMainLoop();
	return true;
}

static void display(void)
{
	sdkStartTimer(&timer);

	if(animate)
	{
		testCuda(&cuda_vbo_resource, imageW, imageH,g_fAnim);
	}

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(translate_x, translate_y,translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// render from the vbo
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glVertexPointer(4, GL_FLOAT, 0, 0);
	glEnableClientState(GL_VERTEX_ARRAY);

	glColor3f(1.0,1.0,1.0);
	if (drawPoints)
    {
		glDrawArrays(GL_POINTS, 0, imageW*imageH);
	}
	else
	{
	//	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBuffer);
		glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		glDrawElements(GL_TRIANGLE_STRIP, ((imageW*2)+2)*(imageH-1), GL_UNSIGNED_INT, 0);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	}
	

	glDisableClientState(GL_VERTEX_ARRAY);
	glClientActiveTexture(GL_TEXTURE0);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glClientActiveTexture(GL_TEXTURE1);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	glutSwapBuffers();
	g_fAnim += 0.01f;
	sdkStopTimer(&timer);
}

static void keyboard(unsigned char key, int x, int y)
{
	switch (key)
	{
	case (27) :
		cleanup();
		cudaDeviceReset();
		exit(EXIT_SUCCESS);
		break;

	case 'p':
		drawPoints = !drawPoints;
		break;

	case ' ':
		animate = !animate;
		break;
	}
}

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_buttons |= 1<<button;
	}

	else if (state == GLUT_UP)
	{
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)  
{
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1)
	{
		rotate_x += dy * 0.2f;
		rotate_y += dx * 0.2f;
	}

	if (mouse_buttons & 2)
	{
		translate_x += dy * 0.01f;
		translate_y += dx * 0.01f;
	}

	else if (mouse_buttons & 4)
	{
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void timerEvent(int value)
{
	glutPostRedisplay();
	glutTimerFunc(0, timerEvent,0);
}

void reshape(int w, int h)
{
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (double) w / (double) h, 0.1, 10.0);

	windowW = w;
	windowH = h;
}

int initGL(int *argc,char **argv)
{
	//cudaGLSetGLDevice(0);
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
	//glutInitWindowPosition(100,100);
	glutInitWindowSize(windowW,windowH);
	glutCreateWindow("CUDA Photometric Stereo");	

	glewInit();
	if (! glewIsSupported("GL_VERSION_2_0 "))
	{
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return false;
	}

	glClearColor(0.0, 0.0,0.0, 1.0);  
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);

	// viewport
	glViewport(0, 0, windowW,windowH);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)windowW/ (GLfloat)windowH, 0.1, 10.0);

	SDK_CHECK_ERROR_GL();
	return true;
}

void createVBO(GLuint *vbo, int size)
{
	assert(vbo);

	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, *vbo);
	glBufferData(GL_ARRAY_BUFFER, size, 0, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	SDK_CHECK_ERROR_GL();

	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, *vbo, cudaGraphicsMapFlagsWriteDiscard);
	
}

void deleteVBO(GLuint *vbo)
{
	if(vbo)
	{
		cudaGraphicsUnregisterResource(cuda_vbo_resource);
		glDeleteBuffers(1, vbo);

		*vbo = NULL;
	}
}

void createMeshIndexBuffer(GLuint *id, int w, int h)
{
	int size = ((w*2)+2)*(h-1)*sizeof(GLuint);

	// create index buffer
	glGenBuffersARB(1, id);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, *id);
	glBufferDataARB(GL_ELEMENT_ARRAY_BUFFER, size, 0, GL_STATIC_DRAW);

	// fill with indices for rendering mesh as triangle strips
	GLuint *indices = (GLuint *) glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (!indices)
	{
		return;
	}

	for (int y=0; y<h-1; y++)
	{
		for (int x=0; x<w; x++)
		{
			*indices++ = y*w+x;
			*indices++ = (y+1)*w+x;
		}

		// start new strip with degenerate triangle
		*indices++ = (y+1)*w+(w-1);
		*indices++ = (y+1)*w;
	}

	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

// create fixed vertex buffer to store mesh vertices
void createMeshPositionVBO(GLuint *id, int w, int h)
{
	createVBO(id, w*h*4*sizeof(float));

	glBindBuffer(GL_ARRAY_BUFFER, *id);
	float *pos = (float *) glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);

	if (!pos)
	{
		return;
	}

	for (int y=0; y<h; y++)
	{
		for (int x=0; x<w; x++)
		{
			float u = x / (float)(w-1);
			float v = y / (float)(h-1);
			*pos++ = u*2.0f-1.0f;
			*pos++ = 0.0f;
			*pos++ = v*2.0f-1.0f;
			*pos++ = 1.0f;
		}
	}

	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void cleanup()
{
	sdkDeleteTimer(&timer);

	if (vbo)
	{
		deleteVBO(&vbo);
	}

	CudaFree();
	free(h_Mask);
	free(h_Src);
	cufftDestroy(pland2zP);
	cufftDestroy(pland2zQ);
	cufftDestroy(planz2d_inv);
	cublasDestroy(handle_Z_min);
	cublasDestroy(handle_Z_max);
}