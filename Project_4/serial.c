/*********************************
High Performance Computing 
Project 4: Suppression and Edge Detection
David Duy Ngo

Changes to base code:
Original image contained in structure, however due to the nature of MPI and for the sake of future projects,
the image buffers will be stored as float pointers instead. So the code has been revamped accordingly.
************************/
#include <stdlib.h>
#include <stdio.h>
#include "image_template.h"
#include <math.h>
#include <sys/time.h>
#include <unistd.h>

/*Part of old code****
typedef struct {
	float* image;
	int width;
	int height;
} Image;
*/

void swapFunc(float* a, float* b){
	float temp = *a;
	*a = *b;
	*b = temp;
}

int compare(const void* a, const void* b) {
	float one = *(float*) a;
	float two = *(float*) b;

	if (one > two)
		return 1;
	else if (two > one)
		return -1;
	return 0;
}

int boundaries(int i, int j, int offset_i, int offset_j, int width, int height){
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

void convolution(int width, int height, float* img, float* kernel, int ker_w, int ker_h, float** output) {
	int i, j, k, l;
	int offseti;
	int offsetj;
	float sum;

	*output = (float*) malloc(sizeof(float)*(height*width));
	
	//first two for-loops are for source image, traversing pixel-by-pixel
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			sum = 0;
			for (k = 0; k < ker_h; k++){
				for (l = 0; l < ker_w; l++){
					offseti = (-1)*floor(ker_h/2)+k;
					offsetj = (-1)*floor(ker_w/2)+l;

					if (boundaries(i, j, offseti, offsetj, width, height) == 1){
						sum = sum + (img[((i+offseti)*width)+(j+offsetj)]*kernel[k*ker_w+l]);	
					}
				}	
			}	
			(*output)[(i*width)+j] = sum;
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
void calculate_magphase(int width, int height, float* horiz, float* vert, float* mag, float* phase){
	int i, j;

	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			mag[i*width + j] = sqrt(pow(vert[i*width +j], 2) + pow(horiz[i*width +j], 2));
            		phase[i*width + j] = atan2(vert[i*width + j], horiz[i*width + j]);
		}	
	}

}

//REMEMBER: pi is M_PI
void suppress(int width, int height, float* suppression, float* magnitude, float* phase) {
	float theta;
	int i, j, offset_i = 0, offset_j = 0, left, right, index;
	memcpy(suppression, magnitude, height*width*sizeof(float));
	
	for (i = 0; i < height; i++){
		for (j = 0; j < width; j++){
			theta = phase[i*width + j];		
			if (theta < 0)
				theta = theta + M_PI;
			theta = (180/M_PI)*theta;
			if (theta <= 22.5 || theta > 157.5){
				offset_i = 0;
				//compare mag with left or right
				offset_j = -1; //left
				index = (i+offset_i)*width + (j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					left = (magnitude[i*width+j] < magnitude[index]);
				offset_j = 1; //right
				if (boundaries(i, j, offset_i, offset_j, width, height))
					right = (magnitude[i*width+j] < magnitude[index]);
				//Suppress[i][j] = 0 if < either
				if (left||right)
					suppression[i*width + j] = 0;			
			}
			else if (theta > 22.5 && theta <= 67.5) {
				//compare with top left and bottom right
				offset_j = -1; offset_i = -1; //top left
				index = (i+offset_i)*width + (j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					left = (magnitude[i*width+j] < magnitude[index]);

				offset_j = 1; offset_i = 1; //bottom right
				index = (i+offset_i)*width + (j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					right = (magnitude[i*width+j] < magnitude[index]);

				//Suppress[i][j] = 0 is less than either
				if (left||right)
					suppression[i*width + j] = 0;			
			}
			else if (theta > 67.5 && theta <= 112.5) {
				//compare with top and bottom
				offset_i = -1; //top
				index = (i+offset_i)*width + (j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					left = (magnitude[i*width+j] < magnitude[index]); //top

				offset_i = 1; //bottom
				index = (i+offset_i)*width + (j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					right = (magnitude[i*width+j] < magnitude[index]);

				//Suppress[i][j] = 0 less than either
				if (left||right)
					suppression[i*width + j] = 0;
			}
			else if (theta > 112.5 && theta <= 157.5) {
				//compare with top right and bottom left
				offset_i = -1; offset_j = 1; //top right
				index = (i+offset_i)*width + (j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					right = (magnitude[i*width+j] < magnitude[index]);

				offset_i = 1; offset_j = -1; //bottom left
				index = (i+offset_i)*width + (j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					left = (magnitude[i*width+j] < magnitude[index]);
				//Suppress[i][j] = 0 if less than either
				if (left||right)
					suppression[i*width + j] = 0;			
			}
		}	
	}
}

void edge(int width, int height, float* suppression, float* hyst, float* edges) {
	int i, j, sum, percent = (0.9*width*height);
	float t_high, t_low;
	int t_right = 0, t_left = 0, top = 0, bottom = 0, b_left = 0, b_right = 0, left = 0, right = 0; //neighbor values
	int offset_i, offset_j, cur_index = 0;
	float *temp = (float*)malloc(sizeof(float)*width*height);
	
	memcpy(temp, suppression, sizeof(float)*width*height);
	qsort(temp, width*height, sizeof(float), compare);
	t_high = temp[percent];
	t_low = t_high/5;
	
	for (i = 0; i < height; i++){
		for (j = 0; j <width; j++){
			if(suppression[i*width+j] >= t_high)
				hyst[i*width+j] = 255;
			else if (suppression[i*width+j] <= t_low)
				hyst[i*width+j] = 0;
			else
				hyst[i*width+j] = 125;
		}
	}
	//calculate edge
	for (i = 0; i < height; i++){
		for( j = 0; j < width; j++){
			if(hyst[i*width+j] == 125) {
				t_right = t_left = top = bottom = b_left = b_right = left = right = 0;
				
				//top three neighbors
				offset_i = -1; offset_j = -1;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					t_left = (hyst[cur_index] == 255); //either a 1 or 0

				offset_i = -1; offset_j = 0;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					top = (hyst[cur_index] == 255); //either a 1 or 0
	
				offset_i = -1; offset_j = 1;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					t_right = (hyst[cur_index] == 255); //either a 1 or 0

				//side two neighbors
				offset_i = 0; offset_j = -1;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					left = (hyst[cur_index] == 255); //either a 1 or 0

				offset_i = 0; offset_j = 1;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					right = (hyst[cur_index] == 255); //either a 1 or 0
			
				//bottom three neighbors
				offset_i = 1; offset_j = -1;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					b_left = (hyst[cur_index] == 255); //either a 1 or 0

				offset_i = 1; offset_j = 0;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					bottom = (hyst[cur_index] == 255); //either a 1 or 0
	
				offset_i = 1; offset_j = 1;
				cur_index = ((i+offset_i)*width)+(j+offset_j);
				if (boundaries(i, j, offset_i, offset_j, width, height))
					b_right = (hyst[cur_index] == 255); //either a 1 or 0

				sum = t_right + t_left + top + bottom + b_left + b_right + left + right; //see if there is a true neighboring pixel
				if (sum >= 1)
					edges[i*width+j] = 255;
				else
					edges[i*width+j] = 0;
			}
			else
				edges[i*width+j] = hyst[i*width+j];
		}	
	}
	free(temp);
}

/**********************
Notable Functions:
void read_image_template(char *name, T **image, int *im_width, int *im_height);
void write_image_template(char *name, T *image, int im_width, int im_height);
***********************/
int main (int argc, char** argv){
	char* file_name;
	char hyst_name[100] = "hyst.pgm";
	char suppress_name[100] = "suppress.pgm";
	char edge_name[100] = "edge.pgm";
	float sigma;
	int size, i, width, height;
	float *kernel, *deriv;
	float *magnitude, *phase;
	float *temp_horiz_image, *temp_vert_image, *ref_image;
	struct timeval start, end;
	
	/*********
	Additional Variables for Project 4
	**********/
	float *suppression, *hyst, *edges;
	//Program 4 end

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

	//flip the derivative
	for (i = 0; i < size/2; i++)
		swapFunc(&deriv[i], &deriv[size-i-1]);

	read_image_template(file_name, &ref_image, &width, &height);

	//allocate memory for like everything
	magnitude = (float*) malloc(sizeof(float)*(width*height));
	phase = (float*) malloc(sizeof(float)*(width*height));
	suppression = (float*) malloc(sizeof(float)*(width*height));
	hyst = (float*) malloc(sizeof(float)*(width*height));
	edges = (float*) malloc(sizeof(float)*(width*height));

	printf("File Name: \"%s\"\nWidth: %d\nHeight %d\n", file_name, width, height);

	/*************
	Horizontal Convolution
	1. make temporary horizontal mask 
	3. Gaussian Derivative
	**************/
	printf("Horizontal Convolve Processing...\n");
	convolution(width, height, ref_image, kernel, 1, size, &temp_horiz_image);
	convolution(width, height, temp_horiz_image, deriv, size, 1, &temp_horiz_image);

	/**************
	Vertical Convolution
	1. make temporary vertical mask 
	3. Gaussian Derivative
	***************/
	printf("Done!\nVertical Convolve Processing...\n");	
	convolution(width, height, ref_image, kernel, size, 1, &temp_vert_image);
	convolution(width, height, temp_vert_image, deriv, 1, size, &temp_vert_image);
	
	printf("Done!\nMangitude/Phase Processing...\n");	
	calculate_magphase(width, height, temp_horiz_image, temp_vert_image, magnitude, phase);
	
	/****Project 4****/
	printf("Done!\nSuppression Processing...\n");	
	suppress(width, height, suppression, magnitude, phase);
	write_image_template<float>(suppress_name, suppression, width, height);

	printf("Done!\nEdge Processing...\n");	
	edge(width, height, suppression, hyst, edges);
	write_image_template<float>(hyst_name, hyst, width, height);
	write_image_template<float>(edge_name, edges, width, height);

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
	free(suppression);
	free(hyst);
	free(edges);

	return 0;
}

