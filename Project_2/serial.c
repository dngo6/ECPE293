/*********************************
High Performance Computing 
Project 2: Finding Horizontal and Vertical Intensity Gradients (Serial)
David Duy Ngo

NOTES: 
*For convolution, treat out of bounds as zeros.

Horizontal Gradient
temp_horizontal = Image convolve (Vertical 1-D Gaussiang_wx1) //smoothen first
Horizontal=temp_horizontal convolve (Horizontal Gaussian derivative1xg_w)//differentiate next

%Remember to flip Gaussian derivative first!
Vertical Gradient
temp_vertical = Image convolve (horizontal 1-D Gaussian1xg_w) %smoothen first
Vertical=temp_vertical convolve (Vertical Gaussian derivativeg_wx1)

Read the input image from the file
Create the Gaussian kernels (horizontal and vertical) and Gaussian derivative kernels (horizontal and vertical)

Loading MPI Module:
module load mpi/openmpi-1.8.8

Compilation:
g++ -std=c99 -lm -o main main.c

./convolution_pthreads  <full path to the image> <floating point value for sigma> <number of pthreads>
************************/
#include <stdlib.h>
#include <stdio.h>
#include "image_template.h"
#include <math.h>
#include <sys/time.h>

typedef struct {
	float* image;
	int width;
	int height;
} Image;

void swapFunc(float* a, float* b){
	float temp = *a;
	*a = *b;
	*b = temp;
}

int boundaries(int i, int j, int offset_i, int offset_j, Image img){
	if (j+offset_j < 0)
		return 0;
	if (i+offset_i < 0)
		return 0;
	if (j+offset_j >= img.width)
		return 0;
	if (i+offset_i >= img.height)
		return 0;

	return 1;
}

void gaussian_kernal(float sigma, float** kernel, int* size) {
	float a = round(2.5*sigma - 0.5);
	int w = 2*a+1;
	float sum = 0;
	int i;

	*kernel = (float*) malloc(sizeof(float)*w);
	
	for (i = 0; i < w; i++){
		(*kernel)[i] = exp(((-1)*(i-a)*(i-a))/(2*sigma*sigma));
		sum = sum+(*kernel)[i];
	}

	for (i = 0; i < w; i++){
		(*kernel)[i] = (*kernel)[i]/sum;
		
	}
	*size = w;
}

void gaussian_deriv(float sigma, float** kernel, int* size) {
	float a = round(2.5*sigma - 0.5);
	int w = 2*a+1;
	float sum = 0;
	int i;
	
	*kernel = (float*)malloc(sizeof(float)*w);
	
	for (i = 0; i < w; i++){
		(*kernel)[i] = -1*(i-a)*exp(((-1)*(i-a)*(i-a))/(2*sigma*sigma));
		sum = sum-(i*(*kernel)[i]);
	}

	for (i = 0; i < w; i++){
		(*kernel)[i] = (*kernel)[i]/sum;
	}
	*size = w;
}

void convolution(Image img, float* kernel, int ker_w, int ker_h, float** output) {
	int i, j, k, l;
	int offseti;
	int offsetj;
	float sum;

	*output = (float*) malloc(sizeof(float)*(img.height*img.width));
	
	//first two for-loops are for source image, traversing pixel-by-pixel
	for (i = 0; i < img.height; i++){
		for (j = 0; j < img.width; j++){
			sum = 0;
			for (k = 0; k < ker_h; k++){
				for (l = 0; l < ker_w; l++){
					offseti = (-1)*floor(ker_h/2)+k;
					offsetj = (-1)*floor(ker_w/2)+l;

					if (boundaries(i,j,offseti,offsetj,img) == 1){
						sum = sum + (img.image[((i+offseti)*img.width)+(j+offsetj)]*kernel[k*ker_w+l]);	
					}
				}	
			}	
			(*output)[(i*img.width)+j] = sum;
		}
	}
}
/**********************
Notable Functions:
void read_image_template(char *name, T **image, int *im_width, int *im_height);
void write_image_template(char *name, T *image, int im_width, int im_height);
***********************/
int main (int argc, char** argv){
	char* file_name;
	char horizontaldGrad[100] = "horizontal_derivative.pgm";
	char verticaldGrad[100] = "vertical_derivative.pgm";
	float sigma;
	int size, i;
	float *kernel, *deriv;
	Image temp_image, ref_image;
	struct timeval start, end;
	
	if (argc != 3){
		printf("The correct argument list is \"./main <file_name> <sigma>\"\n");
		exit(0);	
	}

	file_name = argv[1];
	sigma = atof(argv[2]);
	
	//Start Timer
	gettimeofday(&start, NULL);

	gaussian_kernal(sigma, &kernel, &size);
	gaussian_deriv(sigma, &deriv, &size);
	for (i = 0; i < size/2; i++)
		swapFunc(&deriv[i], &deriv[size-i-1]);

	read_image_template(file_name, &ref_image.image, &ref_image.width, &ref_image.height);
	temp_image.height = ref_image.height;
	temp_image.width = ref_image.width;

	printf("File Name: \"%s\"\nWidth: %d\nHeight %d\n", file_name, ref_image.width, ref_image.height);

	/*************
	Horizontal Convolution
	1. make temporary horizontal mask 
	3. Gaussian Derivative
	**************/
	printf("Horizontal Convolve Processing...\n");
	convolution(ref_image, kernel, 1, size, &temp_image.image);
	//write_image_template<float>(horizontalkGrad, temp_image, ref_image.width, ref_image.height);
	convolution(temp_image, deriv, 1, size, &temp_image.image);
	write_image_template<float>(horizontaldGrad, temp_image.image, ref_image.width, ref_image.height);

	/**************
	Vertical Convolution
	1. make temporary vertical mask 
	3. Gaussian Derivative
	***************/
	printf("Done!\nVertical Convolve Processing...\n");	
	convolution(ref_image, kernel, size, 1, &temp_image.image);
	//write_image_template<float>(verticalkGrad, temp_image, ref_image.width, ref_image.height);
	convolution(temp_image, deriv, size, 1, &temp_image.image);
	write_image_template<float>(verticaldGrad, temp_image.image, ref_image.width, ref_image.height);
	gettimeofday(&end, NULL);
	printf("Done!\nTotal Time to run: %ld milliseconds\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))/1000);
	//End Timer

	//free all
	free(kernel);
	free(deriv);
	free(temp_image.image);
	free(ref_image.image);

	return 0;
}

