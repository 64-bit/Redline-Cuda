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
		void* Data;

		__device__ __host__ unsigned int Index(uint2 position)
		{
			return position.x + position.y * Width;
		}

		__device__ __host__ void SafeWriteColor(uint2 position, Color color)
		{
			if (position.x >= Width || position.y >= Height) 
			{
				return;
			}
			unsigned int index = Index(position);
			Color* asColor = (Color*)Data;
			asColor[index] = color;
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

	};
}