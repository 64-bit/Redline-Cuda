#pragma once

#include <Scene/Components/CameraComponent.h>
#include <Renderer/FrameOutputSettings.h>
#include <memory>
#include <GraphicalResources/Cuda/CudaMesh.h>
#include <GraphicalResources/Bitmap2D.h>
#include <GraphicalResources/Cuda/CudaBitmap2D.h>
#include <glm.h>

namespace Redline
{

	class JankScene
	{
	public:
		unsigned int MeshCount;
		CudaMeshData* Meshes;
		glm::mat4* Transforms;
	};


	class CudaJankFrameRenderer
	{
	private:
		std::shared_ptr<Bitmap2D> _renderTarget;
		std::shared_ptr<CudaBitmap2D> _cudaRenderTarget;

		std::shared_ptr<Redline::Scene> _scene;
		std::shared_ptr<CameraComponent> _camera;
		FrameOutputSettings _frameOutputSettings;
		FrameQuailtySettings _frameQuailtySettings;

		void UploadScene();

		JankScene _jankScene;
		void* d_MeshArray;
		glm::mat4x4* d_TransformsArray;

	public:

		CudaJankFrameRenderer(std::shared_ptr<Redline::Scene>& scene, std::shared_ptr<CameraComponent>& camera,
			const FrameOutputSettings& frameOutputSettings, const FrameQuailtySettings& frameQuailtySettings);

		~CudaJankFrameRenderer();

		std::shared_ptr<Bitmap2D> GetCurrentFrameState();

		void ResetRenderer();

		void RenderFrame();
	};
}