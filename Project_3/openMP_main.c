/*********************************
High Performance Computing 
Project 3: Gradient and Phase Images using Pthreads and OpenMP(Serial)
David Duy Ngo

NOTES: 
*For convolution, treat out of bounds as zeros.

Extend your serial and pthread code from Program 2 to include two simple operations: calculating gradient and phase image. Also develop an OpenMP code with pragmas placed in suitable locations to accelerate the relevant operations using multicore.

In you previous assignment, you obtained the horizontal and vertical gradients of the input image. These can be considered as 'vector' images. The resultant of these vector images gives you the magnitude image and the direction of this resultant is given by the phase image.

Compilation:
g++ -std=c99 -lm -o main main.c

./convolution_pthreads  <full path to the image> <floating point value for sigma> <number of pthreads>
************************/
#include <stdlib.h>
#include <stdio.h>
#include "image_template.h"
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>

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
	#pragma omp parallel for private(j, k, l, sum, offseti, offsetj)
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

/*
To obtain the magnitude and phase images, use the following pseudocode:

for all (i,j) in image do:

      magnitude(i,j)=sqrt( pow(vertical_gradient(i,j),2) + pow(horizontal_gradient(i,j),2));

      phase(i,j)=atan2(vertical_gradient(i,j), horizontal_gradient(i,j));

end for

*/
void calculate_magphase(Image horiz, Image vert, float* mag, float* phase){
	int width = horiz.width, height = horiz.height, i, j;

	#pragma omp parallel for private(j)
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			mag[i*width + j] = sqrt(pow(vert.image[i*width +j], 2) + pow(horiz.image[i*width +j], 2));
            		phase[i*width + j] = atan2(vert.image[i*width + j], horiz.image[i*width + j]);
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
	char magnitude_name[100] = "magnitude.pgm";
	char phase_name[100] = "phase.pgm";
	float sigma;
	int size, i;
	float *kernel, *deriv;
	Image temp_horiz_image, temp_vert_image,ref_image;
	struct timeval start, end;
	
	/*********
	Additional Variables for Project 3
	**********/
	float *magnitude, *phase;
	int num_threads = omp_get_num_procs();
	omp_set_num_threads(num_threads);
	//Program 3 end

	if (argc != 3){
		printf("The correct argument list is \"./main <file_name> <sigma>\"\n");
		exit(0);	
	}
	printf("OpenMP created %d threads...\n", num_threads);

	file_name = argv[1];
	sigma = atof(argv[2]);
	
	//Start Timer
	gettimeofday(&start, NULL);

	gaussian_kernal(sigma, &kernel, &size);
	gaussian_deriv(sigma, &deriv, &size);

	//flip the derivative
	for (i = 0; i < size/2; i++)
		swapFunc(&deriv[i], &deriv[size-i-1]);

	read_image_template(file_name, &ref_image.image, &ref_image.width, &ref_image.height);

	//slight redundancy, but to set up the objects for hor and vert gradients
	temp_horiz_image.height = ref_image.height;
	temp_horiz_image.width = ref_image.width;
	temp_vert_image.height = ref_image.height;
	temp_vert_image.width = ref_image.width;

	//allocate memory for magnitude and phase
	magnitude = (float*) malloc(sizeof(float)*(ref_image.width*ref_image.height));
	phase = (float*) malloc(sizeof(float)*(ref_image.width*ref_image.height));

	printf("File Name: \"%s\"\nWidth: %d\nHeight %d\n", file_name, ref_image.width, ref_image.height);

	/*************
	Horizontal Convolution
	1. make temporary horizontal mask 
	3. Gaussian Derivative
	**************/
	printf("Horizontal Convolve Processing...\n");
	convolution(ref_image, kernel, 1, size, &temp_horiz_image.image);
	convolution(temp_horiz_image, deriv, 1, size, &temp_horiz_image.image);

	/**************
	Vertical Convolution
	1. make temporary vertical mask 
	3. Gaussian Derivative
	***************/
	printf("Done!\nVertical Convolve Processing...\n");	
	convolution(ref_image, kernel, size, 1, &temp_vert_image.image);
	convolution(temp_horiz_image, deriv, size, 1, &temp_vert_image.image);

	
	calculate_magphase(temp_horiz_image, temp_vert_image, magnitude, phase);
	write_image_template<float>(magnitude_name, magnitude, ref_image.width, ref_image.height);
	write_image_template<float>(phase_name, phase, ref_image.width, ref_image.height);

	gettimeofday(&end, NULL);
	printf("Done!\nTotal Time to run: %ld milliseconds\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))/1000);
	//End Timer

	//free all
	free(kernel);
	free(deriv);
	free(magnitude);
	free(phase);
	free(temp_horiz_image.image);
	free(temp_vert_image.image);
	free(ref_image.image);

	return 0;
}

