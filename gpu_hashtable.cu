#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>
#include <stdio.h>

#include "gpu_hashtable.hpp"

__device__ int hash_function1(int key, int table_size, int prim1, int prim2) {
	return ((long)key * prim1) % prim2 % table_size;
}

__device__ int hash_function2(int key, int prim) {
	return prim - (key % prim);
}

__device__ HashElem makeElement(int key, int value) {
	HashElem he = (HashElem) key;
	he = he << 32;
	he += value;
	return he;
}

__device__ int getKey(HashElem he) {
	return he >> 32;
}

__device__ int getValue(HashElem he) {
	HashElem mask = 0xffffffff;
	return he & mask;
}

__device__ void device_insert_hash(HashElem he, HashElem *hash_table, int *nrelem, int prim, int prim1, int prim2, int table_size) {
	int key = getKey(he);
	int index = hash_function1(key, table_size, prim1, prim2);
	int i = index, factor = 1, pas = hash_function2(key, prim), ok = 0;

	while (atomicCAS(&hash_table[i], 0, he) != 0) {
		HashElem he1 = hash_table[i];
		int key1 = getKey(he1);
		if (key == key1) {
			atomicCAS(&hash_table[i], he1, he);
			ok = 1;
			break;
		}
		i = (index + pas * factor) % table_size;
		factor += 1;
	}
	if (!ok)
		atomicAdd(&nrelem[0], 1);
}

__device__ int device_get_values(int key, HashElem *hash_table, int table_size, int prim, int prim1, int prim2) {
	int index = hash_function1(key, table_size, prim1, prim2);
	int searched_key = getKey(hash_table[index]);
	int i = index, factor = 1, pas = hash_function2(key, prim);
	int count = 0;

	while (count < table_size && key != searched_key && hash_table[i] != 0) {
		i = (index + pas * factor) % table_size;
		factor += 1;
		searched_key = getKey(hash_table[i]);
		count += 1;
	}
	if (key == searched_key)
		return getValue(hash_table[i]);
	return 0;
}

__device__ void device_reorder_hashElem(HashElem he, HashElem *hash_table, int table_size, int prim, int prim1, int prim2) {
	if (he > 0) {
		int key = getKey(he);
		int correct_index = hash_function1(key, table_size, prim1, prim2);
		int pas = hash_function2(key, prim);
		int factor = 1, i = correct_index;
		while (atomicCAS(&hash_table[i], 0, he) != 0) {
			i = (correct_index + pas * factor) % table_size;
			factor += 1;
		}
	}
}

__global__ void kernel_insert_hash(int *keys, int* values, int numKeys, HashElem *hash_table,
	int *nrelem, int prim, int prim1, int prim2, int table_size) {

	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
  	for (int j = i; j < numKeys; j += stride) {
		if (keys[j] > 0 && values[j] > 0) {
			HashElem he = makeElement(keys[j], values[j]);
			device_insert_hash(he, hash_table, nrelem, prim, prim1, prim2, table_size);
		}
	}
}

__global__ void kernel_get_values(int *keys, int numKeys, int *values, HashElem *hash_table, int table_size, int prim, int prim1, int prim2) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int j = i; j < numKeys; j += stride) {
		values[j] = device_get_values(keys[j], hash_table, table_size, prim, prim1, prim2);
	}

}

__global__ void kernel_reorder_hashElem(HashElem *hash_table, int table_size, HashElem *new_hash_table, int new_size, int prim, int prim1, int prim2) {
	unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;
	for (int j = i; j < table_size; j += stride) {
		HashElem he = hash_table[j];
		device_reorder_hashElem(he, new_hash_table, new_size, prim, prim1, prim2);
	}
}

unsigned int GpuHashTable::getPrim(unsigned int n) {
	unsigned int i, ok = 0, size = n;
	while(1) {
		for (i = 2; i < size; i++) {
			if (size % i == 0) {
				ok = 1;
				break;
			}
		}
		if (ok == 0 && size >= 2)
			break;
		ok = 0;
		size -= 1;
	}
	return size;
}

/* INIT HASH
 */
GpuHashTable::GpuHashTable(int size) {
	table_size = size * 2;
	table_size = getPrim(table_size);
	blocks_no = table_size / blocks_size;
	if (table_size % blocks_size)
		blocks_no += 1;
	cudaMalloc((void **) &hash_table, table_size * sizeof(HashElem));
	cudaMemset(hash_table, 0, table_size * sizeof(HashElem));
	host_nrelem = (int *) malloc(sizeof(int));
	host_nrelem[0] = 0;
	if (table_size == 2)
		prim = 1;
	else {
		srand(time(NULL));
		int i = rand() % nr_prim;
		prim = primeList[i];
		while (prim >= table_size) {
			i = rand() % nr_prim;
			prim = primeList[i];
		}
	}
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
	table_size = 0;
	cudaFree(hash_table);
	free(host_nrelem);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
	int size = getPrim(numBucketsReshape);
	blocks_no = size / blocks_size;
	if (size % blocks_size)
		blocks_no += 1;
	HashElem *aux_hash_table;
	cudaMalloc((void **) &aux_hash_table, size * sizeof(HashElem));
	cudaMemset(aux_hash_table, 0, size * sizeof(HashElem));
	srand(time(NULL));
	int i = rand() % nr_prim;
	prim = primeList[i];
	while (prim >= size) {
		i = rand() % nr_prim;
		prim = primeList[i];
	}
	kernel_reorder_hashElem<<<blocks_no, blocks_size>>>(hash_table, table_size, aux_hash_table, size, prim, prim1, prim2);
	cudaDeviceSynchronize();
	table_size = size;
	cudaFree(hash_table);
	hash_table = aux_hash_table;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int* values, int numKeys) {
	float loadKeys = (float) numKeys / (float) table_size;
	if (loadKeys + loadFactor() >= 0.80f)
		reshape(2 * table_size);
	int *device_keys, *device_values, *device_nrelem;
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	cudaMalloc((void **) &device_nrelem, sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_nrelem, host_nrelem, sizeof(int), cudaMemcpyHostToDevice);
	kernel_insert_hash<<<blocks_no, blocks_size>>>(device_keys, device_values, numKeys, hash_table, device_nrelem, prim, prim1, prim2, table_size);
	cudaDeviceSynchronize();
	cudaMemcpy(host_nrelem, device_nrelem, sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_keys);
	cudaFree(device_values);
	cudaFree(device_nrelem);
	return false;
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
	int *device_values, *device_keys, *host_values;
	host_values = (int *) malloc(numKeys * sizeof(int));
	cudaMalloc((void **) &device_values, numKeys * sizeof(int));
	cudaMalloc((void **) &device_keys, numKeys * sizeof(int));
	cudaMemcpy(device_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
	kernel_get_values<<<blocks_no, blocks_size>>>(device_keys, numKeys, device_values, hash_table, table_size, prim, prim1, prim2);
	cudaDeviceSynchronize();
	cudaMemcpy(host_values, device_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(device_keys);
	cudaFree(device_values);
	return host_values;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */
float GpuHashTable::loadFactor() {
	return (float) host_nrelem[0] / (float) table_size; // no larger than 1.0f = 100%
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
