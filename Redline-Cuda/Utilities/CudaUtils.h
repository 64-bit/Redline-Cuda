#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <vector>
#include <exception>
#include <stdio.h>

namespace Redline
{

#define cudaChecked(ans) { gpuAssert((ans), __FILE__, __LINE__); }
	inline void gpuAssert(cudaError_t code, const char* file, int line, bool throws = true)
	{
		if (code != cudaSuccess)
		{
			fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
			{
				{
					if (throws)
					{
						throw std::exception();
					}
				}
			}

		}
	}

	class CudaUtils
	{
	public:
		static void InitCuda();

		static void ShutdownCuda();

		template<typename T>
		static T* UploadVector(std::vector<T>& vector);

		template<typename T>
		static T* UploadObject(T& object);

	};

	template<typename T>
	inline T* CudaUtils::UploadVector(std::vector<T>& vector)
	{
		size_t totalSize = sizeof(T) * vector.size();
		T* result;

		cudaChecked(cudaMalloc(&result, totalSize));
		cudaChecked(cudaMemcpy(result, vector.data(), totalSize, cudaMemcpyHostToDevice))
		return result;
	}

	template<typename T>
	inline T* CudaUtils::UploadObject(T& object)
	{
		size_t totalSize = sizeof(T);
		T* result;

		cudaChecked(cudaMalloc(&result, totalSize));
		cudaChecked(cudaMemcpy(result, &object, totalSize, cudaMemcpyHostToDevice))
		return result;
	}
}