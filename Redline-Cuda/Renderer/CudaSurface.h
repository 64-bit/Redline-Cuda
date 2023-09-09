#pragma once
#include <glm.h>

namespace Redline
{
	class CudaSurface
	{
	public:
		glm::mat4 Transform;
		int MeshId;
	};
}