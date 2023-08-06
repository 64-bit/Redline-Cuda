#pragma once

#include <Scene/SceneObjectComponent.h>
#include <mathfu/glsl_mappings.h>
#include "../../../GraphicalResources/Material.h"
#include <GraphicalResources/Cuda/CudaMesh.h>

namespace Redline
{
	class CudaMeshRenderer : public SceneObjectComponent
	{
	public:
		CudaMeshRenderer(SceneObject& owner);

		std::shared_ptr<CudaMesh> Mesh;
		std::vector<std::shared_ptr<Material>> SurfaceMaterials;
	};
}
