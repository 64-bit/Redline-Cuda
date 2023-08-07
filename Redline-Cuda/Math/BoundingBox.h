#pragma once
#include "../GraphicalResources/MeshVertex.h"
#include <vector>
#include "../GraphicalResources/MeshTriangle.h"
#include "cuda_runtime.h"
#include <Math/Ray.h>
#include <glm.h>

namespace Redline
{

	class BoundingBox
	{
	public:
		glm::vec3 Min;
		glm::vec3 Max;

		BoundingBox();

		__host__ __device__ bool DoesRayIntersect(const Ray& ray, float& outTNear, float& outTFar)
		{

			float tNear = -FLT_MAX;
			float tFar = FLT_MAX;


			for (int i = 0; i < 3; i++)
			{
				float t1 = (Min[i] - ray.Origin[i]) / ray.Direction[i];
				float t2 = (Max[i] - ray.Origin[i]) / ray.Direction[i];

				if (t1 > t2)
				{
					float temp = t1;
					t1 = t2;
					t2 = temp;
				}

				if (t1 > tNear)
				{
					tNear = t1;
				}

				if (t2 < tFar)
				{
					tFar = t2;
				}
			}

			if (tNear > tFar
				|| tFar < 0.0f)
			{
				return false;
			}

			outTNear = tNear;
			outTFar = tFar;

			return true;
		}

		void EnlargeByPoints(const std::vector<MeshVertex>& verticies);
		void EnlargeByPoint(const glm::vec3& point);
		void EnlargeByPoint(const mathfu::vec3& point);

		void EnlargeByTriangles(const std::vector<MeshTriangle>& triangles, const std::vector<MeshVertex>& verticies);

		void EnlargeByBounds(const BoundingBox& otherBounds);

		glm::vec3 GetCenter() const;

	private:
	};
}
