/*
David Duy Ngo
MPI TEST PROGRAM:
	COMPILE:
	mpicxx mpi.c -o test

	EXECUTE:
	mpiexec -n 4 ./test

Below is a sample program that I wrote to demonstrate MPI Scatter and Gather.
The master process broadcasts the array size (in this case, the height doesn't matter) to other processes.
The sub processes add every element in their respective arrays by 1 and the master process gathers the new values.

I did not write this program to be scalable for any number of processes other than 4 but you can play around with it.

The program below showcases essentially pretty much every MPI function needed to write Program 4's main function in their exact order.
However, I did not take account ghosting yet using SendRecv functions.

*/
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>

void addVal(int *arr, int size, int rank) {
	int i;
	for (i = 0; i < size; i++) {
		arr[i]++;	
	}
	printf("Proc %d, %d %d %d %d\n", rank, arr[0], arr[1], arr[2], arr[3]);
}

int main() {
	int num_proc, i, num, rank;
	int width = 4, height = 4;
	int array[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}; //sort of simulates the image array but don't expect all processes in our HPC program to have this info.
	int *newArr;

	MPI_Init(NULL, NULL);
	//number of processes
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	//rank of processes, unique to each process.
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	if (rank == 0)
		printf("%d Processers\n", num_proc);

	newArr = (int*) malloc(sizeof(int)*width);

	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD); //Width
	MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD); //Height
	MPI_Scatter(array, 4, MPI_INT, newArr, 4, MPI_INT, 0, MPI_COMM_WORLD); //scatter the array. Each process has a chunk of size 4.

	printf("Processor: %d says width = %d and height = %d\n\n", rank, width, height);

	addVal(newArr, 4, rank);
	
	MPI_Gather(newArr, 4, MPI_INT, array, 4, MPI_INT, 0, MPI_COMM_WORLD); //idk how but everything is in the right order everytime.

	if (rank == 0) {
		for (i = 0; i < 16; i++)
			printf("%d ", array[i]);
		printf("\n");	
	}

	MPI_Finalize();
	free(newArr);
	return 0;
}
