#include "CudaMeshRenderer.h"

Redline::CudaMeshRenderer::CudaMeshRenderer(SceneObject& owner)
	: SceneObjectComponent(owner)

{
	Mesh = nullptr;
}
