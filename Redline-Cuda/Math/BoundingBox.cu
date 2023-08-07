#include "BoundingBox.h"
#include "Ray.h"

using namespace Redline;
using namespace glm;

BoundingBox::BoundingBox()
{
	Max = vec3(-INFINITY, -INFINITY, -INFINITY);
	Min = vec3(INFINITY, INFINITY, INFINITY);
}


glm::vec3 Make3(mathfu::vec3 i) 
{
	return glm::vec3(i.x, i.y, i.z);
}

void Redline::BoundingBox::EnlargeByPoints(const std::vector<MeshVertex>& verticies)
{
	//Pick any vertex, and set that as the inital min and max
	if (verticies.size() > 0)
	{

		Min = Make3(verticies[0].Position);
		Max = Min;

		for (auto& vertex : verticies)
		{
			Min = min(Min, Make3(vertex.Position));
			Max = max(Max, Make3(vertex.Position));
		}
	}
	else
	{
		Min = vec3(0.0f, 0.0f, 0.0f);
		Max = Min;
	}
}

void BoundingBox::EnlargeByPoint(const mathfu::vec3& point)
{
	EnlargeByPoint(Make3(point));
}

void BoundingBox::EnlargeByPoint(const glm::vec3& point)
{
	Min = min(Min, point);
	Max = max(Max, point);
}

void BoundingBox::EnlargeByTriangles(const std::vector<MeshTriangle>& triangles, const std::vector<MeshVertex>& verticies)
{
	for(auto& triangle : triangles)
	{
		auto& A = Make3(verticies[triangle.A].Position);
		auto& B = Make3(verticies[triangle.B].Position);
		auto& C = Make3(verticies[triangle.C].Position);

		Min = min(Min, A);
		Min = min(Min, B);
		Min = min(Min, C);

		Max = max(Max, A);
		Max = max(Max, B);
		Max = max(Max, C);
	} 
}

void Redline::BoundingBox::EnlargeByBounds(const BoundingBox & otherBounds)
{
	Min = min(Min, otherBounds.Min);
	Max = max(Max, otherBounds.Max);
}

vec3 BoundingBox::GetCenter() const
{
	return (Min + Max) * 0.5f;
}
