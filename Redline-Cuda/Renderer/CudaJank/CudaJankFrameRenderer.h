#pragma once

#include <Scene/Components/CameraComponent.h>
#include <Renderer/FrameOutputSettings.h>
#include <memory>
#include <GraphicalResources/Cuda/CudaMesh.h>
#include <GraphicalResources/Bitmap2D.h>
#include <GraphicalResources/Cuda/CudaBitmap2D.h>
#include <glm.h>
#include <Scene/CompiledScene.h>

namespace Redline
{
	class CompiledScene;

	class CudaJankFrameRenderer
	{
	private:
		std::shared_ptr<Bitmap2D> _renderTarget;
		std::shared_ptr<CudaBitmap2D> _cudaRenderTarget;

		std::shared_ptr<Redline::Scene> _scene;
		std::shared_ptr<CameraComponent> _camera;
		FrameOutputSettings _frameOutputSettings;
		FrameQuailtySettings _frameQuailtySettings;

		CompiledScene _compiledScene;

	public:

		CudaJankFrameRenderer(std::shared_ptr<Redline::Scene>& scene, std::shared_ptr<CameraComponent>& camera,
			const FrameOutputSettings& frameOutputSettings, const FrameQuailtySettings& frameQuailtySettings);

		~CudaJankFrameRenderer();

		std::shared_ptr<CudaBitmap2D> GetCurrentFrameState();

		void ResetRenderer();

		void RenderFrame();
	};
}