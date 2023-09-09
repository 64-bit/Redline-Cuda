#pragma once
#include <memory>
#include "cuda_runtime.h"
#include <GraphicalResources/Color.h>

namespace Redline
{

	class Bitmap2D;

	struct CudaBitmapData
	{
		unsigned int Width;
		unsigned int Height;
		Color* Data;

		__device__ __host__ __forceinline inline size_t SizeInBytes() const
		{
			return sizeof(Color) * Width * Height;
		}

		__device__ __host__ __forceinline inline unsigned int Index(uint2 position) const
		{
			return position.x + position.y * Width;
		}

		__device__ __forceinline inline Color& operator[](uint2 position)
		{ 
			int index = Index(position);
			return Data[index];
		}

		__device__ __forceinline inline const Color& operator[](uint2 position) const
		{
			int index = Index(position);
			return Data[index];
		}

		__device__ __host__ __forceinline inline void SafeWriteColor(uint2 position, Color color)
		{
			if (position.x >= Width || position.y >= Height) 
			{
				return;
			}
			unsigned int index = Index(position);
			Data[index] = color;
		}

		__device__ __forceinline inline void Write(uint2 position, Color color) 
		{
			unsigned int index = Index(position);
			Data[index] = color;
		}

		__device__ __forceinline inline Color Read(uint2 position)
		{
			unsigned int index = Index(position);
			return Data[index];
		}


	};

	class CudaBitmap2D
	{
	private:
		size_t _dataSize;

	public:
		CudaBitmapData Data;
		void* DataPointer;

		CudaBitmap2D(unsigned int width, unsigned int height);
		~CudaBitmap2D();

		void CopyToBitmap2D(std::shared_ptr<Bitmap2D> destination);

		void CudaClear(Color color);

		void ComputeDispatchSize(dim3& gridsOut, dim3& blocksOut);

		void RotateForDisplay();

	};
}