#include "CudaBitmap2D.h"
#include <Utilities/CudaUtils.h>
#include <GraphicalResources/Bitmap2D.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace Redline;



__device__ __forceinline __forceinline__ inline uint2 computeThreadIndex2D() 
{
	uint2 result;
	result.x = threadIdx.x + blockIdx.x * blockDim.x;
	result.y = threadIdx.y + blockIdx.y * blockDim.y;
	return result;
}

__global__ void rotatekernel(CudaBitmapData data) 
{
	uint2 i = computeThreadIndex2D();

	if (i.x > data.Width || i.y > data.Height) 
	{
		return;
	}

	data[i] = data[i].RotateForSDLDisplay();
}

void Redline::CudaBitmap2D::RotateForDisplay()
{
	dim3 grid, block;
	ComputeDispatchSize(grid, block);

	rotatekernel << <grid, block >> > (Data);
	cudaChecked(cudaGetLastError());
}

__global__ void clearKernel(CudaBitmapData data, Color clearColor)
{
	uint2 i = computeThreadIndex2D();
	if (i.x > data.Width || i.y > data.Height)
	{
		return;
	}

	data[i] = clearColor;
}


Redline::CudaBitmap2D::CudaBitmap2D(unsigned int width, unsigned int height)
{
	_dataSize = sizeof(char) * 4 * width * height;
	cudaChecked(cudaMalloc(&DataPointer, _dataSize));

	Data = CudaBitmapData();
	Data.Data = (Color*)DataPointer;
	Data.Width = width;
	Data.Height = height;
}

Redline::CudaBitmap2D::~CudaBitmap2D()
{
	if (DataPointer != nullptr)
	{
		cudaChecked(cudaFree(DataPointer));
	}
	DataPointer = nullptr;
}

void Redline::CudaBitmap2D::CopyToBitmap2D(std::shared_ptr<Bitmap2D> destination)
{
	void* copyDest = destination->Pixels;
	if (destination->Width != Data.Width) 
	{
		throw std::exception();
	}
	if (destination->Height != Data.Height)
	{
		throw std::exception();
	}


	//cudaChecked(cudaDeviceSynchronize());
	cudaChecked(cudaMemcpy(copyDest, DataPointer, _dataSize, cudaMemcpyDeviceToHost));
}

void Redline::CudaBitmap2D::CudaClear(Color color)
{
	dim3 block;// (8, 8);

	//int gridx = (Data.Width / 8) + ((Data.Width % 8) > 0 ? 1 : 0);
	//i/nt gridy = (Data.Height / 8) + ((Data.Height % 8) > 0 ? 1 : 0);

	dim3 grid;// (gridx, gridy);

	ComputeDispatchSize(grid, block);

	clearKernel<<<grid, block >>>(Data, color);
	cudaChecked(cudaGetLastError());

}

void Redline::CudaBitmap2D::ComputeDispatchSize(dim3& gridsOut, dim3& blocksOut)
{
	blocksOut = dim3(8, 8);

	int gridx = (Data.Width / 8) + ((Data.Width % 8) > 0 ? 1 : 0);
	int gridy = (Data.Height / 8) + ((Data.Height % 8) > 0 ? 1 : 0);

	gridsOut = dim3(gridx, gridy);
}

