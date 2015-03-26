////////////////////////////////////////////////////////////////////////////
//////////////////////   Load pictures  :  C   ///////////////////////////////
////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#   pragma warning( disable : 4996 ) // disable deprecated warning 
#endif

#pragma pack(1)

typedef struct
{
	short type;
	int size;
	short reserved1;
	short reserved2;
	int offset;
} BMPHeader;

typedef struct
{
	int size;
	int width;
	int height;
	short planes;
	short bitsPerPixel;
	unsigned compression;
	unsigned imageSize;
	int xPelsPerMeter;
	int yPelsPerMeter;
	int clrUsed;
	int clrImportant;
} BMPInfoHeader;

typedef struct
{
	unsigned char x;
} uchar1;

extern "C" void LoadBMPFile(unsigned char  **dst, int *offset,int *width, int *height, const char *name)
{	
	BMPHeader hdr;
	BMPInfoHeader infoHdr;
	int x,y;

	FILE *fd;

	printf("Loading %s...\n", name);

	if (!(fd = fopen(name,"rb")))
	{
		printf("***BMP load error: file access denied***\n");
		exit(EXIT_SUCCESS);
	}

	fread(&hdr, sizeof(hdr), 1, fd);
	fread(&infoHdr, sizeof(infoHdr), 1, fd);
	
	if (hdr.type != 0x4d42)		
	{
		printf("It's not a BMP file! \n");
		exit(0);
	}

	if (infoHdr.compression)
	{
		printf("***BMP load error: compressed image***\n");
		exit(EXIT_SUCCESS);
	}

	*width  = infoHdr.width;  //1280
	*height = infoHdr.height;	//1024
	*offset = hdr.offset;

	int lineByte = (*width *infoHdr.bitsPerPixel/8+3)/4*4;

	 fseek(fd, hdr.offset, SEEK_SET);
	//printf("%d\n%d\n%d\n",hdr.offset,sizeof(hdr),sizeof(infoHdr));//,sizeof(RGBQUAD),sizeof(RGBQUAD)*256);
	 *dst   = (unsigned char *)malloc(*width  * *height *sizeof(unsigned char));

	for (y = 0; y < infoHdr.height; y++)
	{
		for (x = 0; x < infoHdr.width; x++)
		{
			(*dst)[(y * infoHdr.width + x)] =(unsigned char) fgetc(fd);
		}
		for (x = 0; x < (4 - (3 * infoHdr.width) % 4) % 4; x++)
			fgetc(fd);
	}

	if (ferror(fd))
	{
		printf("***Unknown BMP load error.***\n");
		free(*dst);
		exit(EXIT_SUCCESS);
	}
	else
		printf("BMP file loaded successfully!\n");
	
	fclose(fd);
}

extern "C" void LoadManyBmp( unsigned char **dst, int offset, int width, int height, char **nameList)
{
	int x, y;
	FILE *fd;
	*dst   = (unsigned char *)malloc(width *height*6*sizeof(unsigned char));

	for(int i = 0; i < 6; i++ )
	{
		if (!(fd = fopen(nameList[i],"rb")))
		{
			printf("***BMP load error: file access denied***\n");
			exit(EXIT_SUCCESS);
		}

		fseek(fd, offset, SEEK_SET);

		for (y = 0; y < height; y++)
		{
			for (x = 0; x <width; x++)
				(*dst)[width*height*i + y * width + x]=(unsigned char)fgetc(fd);		

			for (x = 0; x < (4 - (3 * width) % 4) % 4; x++)
			fgetc(fd);
		}		
	}
		
		if (ferror(fd))
		{
			printf("***Unknown BMP load error.***\n");
			free(*dst);
			exit(EXIT_SUCCESS);
		}
		else
			printf("BMP file loaded successfully!\n");
	fclose(fd);
}

