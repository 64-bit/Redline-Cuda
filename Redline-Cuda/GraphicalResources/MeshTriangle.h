#pragma once
#include "MeshVertex.h"

namespace Redline
{
	class BoundingBox;

	struct MeshTriangle
	{
		unsigned A;
		unsigned B;
		unsigned C;
		unsigned Material;
	};

	struct BVHTriangle
	{
		MeshVertex A;
		MeshVertex B;
		MeshVertex C;
		unsigned Material;

		BoundingBox GetBounds();
	};

	struct BVHTriangle_Vertex
	{
		mathfu::vec3 Verts[3]; //9

		unsigned Material; //10



		//
	};

	struct BVHTriangle_Properties
	{
		mathfu::vec2 UV[3];
		mathfu::vec3 Normal[3];
		mathfu::vec3 BiNormal[3];
		mathfu::vec3 Tangent[3];
	};


}
