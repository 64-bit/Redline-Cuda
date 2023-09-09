#pragma once
#include "cuda_runtime.h"
#include<memory>

namespace Redline
{
	class Scene;
	class CudaMesh;
	class CudaSurface;

	//This represents a scene that has been compiled and optimized into structures for rendering
	class CompiledScene
	{
	public:
		__host__ CompiledScene();
		__host__ CompiledScene(std::shared_ptr<Scene> scene);
		__host__ void Dispose();

		//Surfaces
		CudaSurface* Surfaces;
		int SurfaceCount;

		//Top level Acceleration structs

		//Textures, Materials, Meshes
		CudaMesh* Meshes;

		//Lights

		//Camera info ?? (no this will pass in separately)
	};
}