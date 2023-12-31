#pragma once

#include <string>
#include <vector>
#include "cuda_runtime.h"
#include "mathfu/glsl_mappings.h"
#include <Math/BoundingBox.h>

namespace objl
{
	struct Mesh;
	class Loader;
}

namespace tinygltf
{
	struct Primitive;
	struct Mesh;
	class Model;
}

namespace Redline
{
	class CudaMeshBuilder;

	class CudaMesh
	{
	public:
		unsigned int TriangleCount;
		unsigned int VertexCount;

		BoundingBox Bounds;

		void* Verticies;
		uint4* Triangles;
		void* Uv0;
		void* Normals;
		void* BiTangents;
		void* Tangents;

		CudaMesh();

		CudaMesh(CudaMeshBuilder& source);

		void Dispose();
	};

	class CudaMeshBuilder
	{
	private:



	public:
		std::string Name;
		float BoundingRadius;

		std::vector<uint4> Triangles;

		std::vector<mathfu::vec3> Vertices;
		std::vector<mathfu::vec2> Uv0;

		std::vector<mathfu::vec3> Normals;
		std::vector<mathfu::vec3> BiTangents;
		std::vector<mathfu::vec3> Tangents;

		CudaMesh _cudaMeshData;

		BoundingBox Bounds;


		unsigned int GetTriangleCount() const;

		CudaMeshBuilder(const std::string& name, tinygltf::Mesh& mesh, tinygltf::Model& gltfFile);
		~CudaMeshBuilder();

	private:
		void ComputeTangents();

		void ComputeBounds();

		void ComputeTangentBitangent(uint4 triangle, mathfu::vec3& tangent, mathfu::vec3& biTangent);

		void AppendGLTFPrimitive(tinygltf::Primitive& primitive, tinygltf::Model& gltfFile, int primitiveID);
	};
}