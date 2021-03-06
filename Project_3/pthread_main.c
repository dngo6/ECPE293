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
#include <pthread.h>

typedef struct {
	int thread_id;
	int ker_w;
	int ker_h;
} thread_data;

//global variables
int num_threads, width, height;
float *kernel, *deriv, sigma;
float *ref_image;
float *temp_horiz_image, *temp_vert_image;

/*********
Additional Variables for Project 3
**********/
float *magnitude, *phase;

void swapFunc(float* a, float* b){
	float temp = *a;
	*a = *b;
	*b = temp;
}

int boundaries(int i, int j, int offset_i, int offset_j, int width, int height, int start){
	if (j+offset_j < 0)
		return 0;
	if (i+offset_i < 0)
		return 0;
	if (j+offset_j >= width)
		return 0;
	if (i+offset_i >= height)
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

void convolution(float* img, int width, int height, int start, int end, float* kernel, int ker_w, int ker_h, float** output) {
	int i, j, k, l;
	int offseti;
	int offsetj;
	float sum;
	
	//first two for-loops are for source image, traversing pixel-by-pixel
	for (i = start; i < end; i++){
		for (j = 0; j < width; j++){
			sum = 0;
			for (k = 0; k < ker_h; k++){
				for (l = 0; l < ker_w; l++){
					offseti = (-1)*floor(ker_h/2)+k;
					offsetj = (-1)*floor(ker_w/2)+l;

					if (boundaries(i, j, offseti, offsetj, width, end, start) == 1){
						sum = sum + (img[((i+offseti)*width)+(j+offsetj)]*kernel[k*ker_w+l]);	
					}
				}	
			}
			(*output)[(i*width)+j] = sum;
		}
	}
}


void calculate_magphase(float* horiz, float* vert, float* mag, float* phase, int start, int end){
	int i, j;

	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			mag[i*width + j] = sqrt(pow(vert[i*width +j], 2) + pow(horiz[i*width +j], 2));
            		phase[i*width + j] = atan2(vert[i*width + j], horiz[i*width + j]);
		}	
	}

}

void* thread_magphase(void* thread_args){
	thread_data *data; 
	data = (thread_data *) thread_args;
	int start_index = data->thread_id*((height)/num_threads);
	int end_index = start_index+((height)/num_threads);

	printf("Thread #%d | Start:%d | End:%d\n", data->thread_id, start_index, end_index);
	
	calculate_magphase(temp_horiz_image, temp_vert_image, magnitude, phase, start_index, end_index);

	pthread_exit(NULL);
}

void* thread_func_vert(void* thread_args){
	thread_data *data; 
	data = (thread_data *) thread_args;
	int start_index = data->thread_id*((height)/num_threads);
	int end_index = start_index+((height)/num_threads);

	printf("Thread #%d | Start:%d | End:%d\n", data->thread_id, start_index, end_index);
	
	convolution(ref_image, width, height, start_index, end_index, kernel, data->ker_w, data->ker_h, &temp_vert_image);
	convolution(ref_image, width, height, start_index, end_index, deriv, data->ker_w, data->ker_h, &temp_vert_image);

	pthread_exit(NULL);
}

void* thread_func_horiz(void* thread_args){
	thread_data *data; 
	data = (thread_data *) thread_args;
	int start_index = data->thread_id*((height)/num_threads);
	int end_index = start_index+((height)/num_threads);

	printf("Thread #%d | Start:%d | End:%d\n", data->thread_id, start_index, end_index);
	
	convolution(ref_image, width, height, start_index, end_index, kernel, data->ker_w, data->ker_h, &temp_horiz_image);
	convolution(ref_image, width, height, start_index, end_index, deriv, data->ker_w, data->ker_h, &temp_horiz_image);

	pthread_exit(NULL);
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
	int size, i;
	pthread_t *threads;
	thread_data *data;
	struct timeval start, end;
	
	//Program 3 end

	if (argc != 4){
		printf("The correct argument list is \"./main <file_name> <sigma> <num_threads>\"\n");
		exit(0);	
	}

	file_name = argv[1];
	sigma = atof(argv[2]);
	num_threads = atoi(argv[3]);	
	
	threads = (pthread_t*) malloc(sizeof(pthread_t)*num_threads);
	data = (thread_data*) malloc(sizeof(thread_data)*num_threads);

	//Start Timer
	gettimeofday(&start, NULL);

	gaussian_kernal(sigma, &kernel, &size);
	gaussian_deriv(sigma, &deriv, &size);

	//flip the derivative
	for (i = 0; i < size/2; i++)
		swapFunc(&deriv[i], &deriv[size-i-1]);

	read_image_template(file_name, &ref_image, &width, &height);

	//allocate memory for magnitude and phase
	temp_horiz_image = (float*) malloc(sizeof(float)*(height*width));
	temp_vert_image = (float*) malloc(sizeof(float)*(height*width));
	magnitude = (float*) malloc(sizeof(float)*(width*height));
	phase = (float*) malloc(sizeof(float)*(width*height));

	printf("File Name: \"%s\"\nWidth: %d\nHeight %d\n", file_name, width, height);

	/*************
	Horizontal Convolution
	1. make temporary horizontal mask 
	3. Gaussian Derivative
	**************/
	printf("Horizontal Convolve Processing...\n");
	
	for (i = 0; i < num_threads; i++){
		data[i].thread_id = i;
		data[i].ker_w = size;
		data[i].ker_h = 1;
		pthread_create(&threads[i], NULL, thread_func_horiz, (void *) &data[i]);	
	}
	
	for (i = 0; i < num_threads; i++)
		pthread_join(threads[i], NULL);

	/**************
	Vertical Convolution
	1. make temporary vertical mask 
	3. Gaussian Derivative
	***************/
	
	printf("Done!\nVertical Convolve Processing...\n");	
	
	for (i = 0; i < num_threads; i++){
		data[i].thread_id = i;
		data[i].ker_w = 1;
		data[i].ker_h = size;
		pthread_create(&threads[i], NULL, thread_func_vert, (void *) &data[i]);	
	}
	
	
	for (i = 0; i < num_threads; i++)
		pthread_join(threads[i], NULL);

	/************
	Magphase stage
	**************/	
	printf("Done!\nCalculating Magphase...\n");
	for (i = 0; i < num_threads; i++){
		data[i].thread_id = i;
		pthread_create(&threads[i], NULL, thread_magphase, (void *) &data[i]);	
	}
	
	
	for (i = 0; i < num_threads; i++)
		pthread_join(threads[i], NULL);

	write_image_template<float>(magnitude_name, magnitude, width, height);
	write_image_template<float>(phase_name, phase, width, height);
	
	gettimeofday(&end, NULL);
	printf("Done!\nTotal Time to run: %ld milliseconds\n", ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec))/1000);
	//End Timer

	//free all
	free(kernel);
	free(deriv);
	free(magnitude);
	free(phase);
	free(temp_horiz_image);
	free(temp_vert_image);
	free(ref_image);
	free(threads);
	free(data);

	return 0;
}

