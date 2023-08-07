#include "CudaMesh.h"
#include <tiny_gltf.h>
#include <FileFormats/GLTFUtilities.h>
#include <Utilities/CudaUtils.h>

using namespace std;
using namespace Redline;
using namespace mathfu;

unsigned int Redline::CudaMesh::GetTriangleCount() const
{
	return Triangles.size();
}

Redline::CudaMesh::CudaMesh(const std::string& name, tinygltf::Mesh& mesh, tinygltf::Model& gltfFile)
{
	Name = name;
	BoundingRadius = 0.0f;
	memset(&_cudaMeshData, 0, sizeof(CudaMeshData));

	int primitiveID = 0;
	for (auto& primitive : mesh.primitives)
	{
		AppendGLTFPrimitive(primitive, gltfFile, primitiveID++);
	}

	ComputeTangents();
	ComputeBounds();

	_cudaMeshData = CudaMeshData(*this);
	printf("Loaded mesh %s with %zu vertices and %u triangles\n", Name.c_str(), Vertices.size(), GetTriangleCount());
}

Redline::CudaMesh::~CudaMesh()
{
	_cudaMeshData.Dispose();
}

void Redline::CudaMesh::ComputeTangentBitangent(uint4 triangle, vec3& tangent, vec3& biTangent)
{
	vec3 positionA = Vertices[triangle.x];
	vec3 positionB = Vertices[triangle.y];
	vec3 positionC = Vertices[triangle.z];

	vec2 uvA = Uv0[triangle.x];
	vec2 uvB = Uv0[triangle.y];
	vec2 uvC = Uv0[triangle.z];

	vec3 pd1 = positionB - positionA;
	vec3 pd2 = positionC - positionA;


	vec2 d1 = uvB - uvA;
	vec2 d2 = uvC - uvA;

	vec3 d13 = vec3(d1, 0.0f);
	vec3 d23 = vec3(d2, 0.0f);


	float det = (d1.x * d2.y) - (d1.y * d2.x);
	det = 1.0f / det;

	tangent.x = det * (d2.y * pd1.x + -d1.y * pd2.x);
	tangent.y = det * (d2.y * pd1.y + -d1.y * pd2.y);
	tangent.z = det * (d2.y * pd1.z + -d1.y * pd2.z);

	biTangent.x = det * (-d2.x * pd1.x + d1.x * pd2.x);
	biTangent.y = det * (-d2.x * pd1.y + d1.x * pd2.y);
	biTangent.z = det * (-d2.x * pd1.z + d1.x * pd2.z);

	BiTangents[triangle.x] += biTangent;
	Tangents[triangle.x] += tangent;

	BiTangents[triangle.y] += biTangent;
	Tangents[triangle.y] += tangent;

	BiTangents[triangle.z] += biTangent;
	Tangents[triangle.z] += tangent;
}

void Redline::CudaMesh::ComputeTangents()
{
	//Init binormal and tanget
	for (int i = 0; i < Normals.size(); i++) 
	{
		BiTangents.push_back(vec3(0.0f));
		Tangents.push_back(vec3(0.0f));
	}

	//For each triangle, compute tangent and accumulate, we normalise this later.
	for (const uint4& triangle : Triangles)
	{
		vec3 tangent, biNormal;
		ComputeTangentBitangent(triangle, tangent, biNormal);
	}

	//For each vertex, normalize tangent and bitangent. also place a copy of the position into the positions only vector
	for (int i = 0; i < Vertices.size(); i++)
	{
		const auto baseT = Tangents[i];
		const auto vertexNormal = Normals[i];
		const auto vertexBiTangent = BiTangents[i];

		const auto orthorNormTangent = (baseT - vertexNormal * vec3::DotProduct(vertexNormal, baseT)).Normalized();

		float handedness = vec3::DotProduct(vec3::CrossProduct(vertexNormal, baseT), vertexBiTangent) < 0.0f ? -1.0f : 1.0f;

		BiTangents[i] = vec3::CrossProduct(vertexNormal, orthorNormTangent) * handedness;
		Tangents[i] = orthorNormTangent;
	}
}

void Redline::CudaMesh::ComputeBounds()
{
	Bounds = BoundingBox();
	for (vec3& vert : Vertices) 
	{
		Bounds.EnlargeByPoint(vert);
	}
}

void Redline::CudaMesh::AppendGLTFPrimitive(tinygltf::Primitive& primitive, tinygltf::Model& gltfFile, int primitiveID)
{
	unsigned int vertexStartPosition = Vertices.size();

	auto& indiciesAccessor = gltfFile.accessors[primitive.indices];
	vector<int> indicies;
	GLTFUtilities::ReadGLTFAccessorToIntVector(indiciesAccessor, gltfFile, indicies);

	//Position
	if (primitive.attributes.find("POSITION") != primitive.attributes.end())
	{
		auto& positionAccessor = gltfFile.accessors[primitive.attributes["POSITION"]];
		GLTFUtilities::ReadGLTFAccessorToVectorVector<vec3>(positionAccessor, gltfFile, Vertices);
	}
	//UV0
	if (primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end())
	{
		auto& uvAccessor = gltfFile.accessors[primitive.attributes["TEXCOORD_0"]];
		GLTFUtilities::ReadGLTFAccessorToVectorVector<vec2>(uvAccessor, gltfFile, Uv0);
	}
	//Normals
	if (primitive.attributes.find("NORMAL") != primitive.attributes.end())
	{
		auto& normalAccessor = gltfFile.accessors[primitive.attributes["NORMAL"]];
		GLTFUtilities::ReadGLTFAccessorToVectorVector<vec3>(normalAccessor, gltfFile, Normals);
	}



	//Ensure we have all elements
	if (Vertices.size() != Uv0.size()
		|| Vertices.size() != Normals.size())
	{
		throw std::exception();
		return;//ERROR
	}

	//construct triangles
	for (auto i = 0; i < indicies.size() / 3; i++)
	{
		uint4 triangle;
		triangle.x = indicies[i * 3 + 0] + vertexStartPosition;
		triangle.y = indicies[i * 3 + 1] + vertexStartPosition;
		triangle.z = indicies[i * 3 + 2] + vertexStartPosition;
		triangle.w = primitiveID;
		Triangles.push_back(triangle);
	}
}

Redline::CudaMeshData::CudaMeshData()
{
	VertexCount = 0;
	TriangleCount = 0;

	Triangles = nullptr;
	Verticies = nullptr;
	Uv0 = nullptr;
	Normals = nullptr;
	BiTangents = nullptr;
	Tangents = nullptr;
}

Redline::CudaMeshData::CudaMeshData(CudaMesh& source)
{
	VertexCount = source.Vertices.size();
	TriangleCount = source.Triangles.size();

	Triangles = CudaUtils::UploadVector(source.Triangles);
	Verticies = CudaUtils::UploadVector(source.Vertices);
	Uv0 = CudaUtils::UploadVector(source.Uv0);
	Normals = CudaUtils::UploadVector(source.Normals);
	BiTangents = CudaUtils::UploadVector(source.BiTangents);
	Tangents = CudaUtils::UploadVector(source.Tangents);

	Bounds = source.Bounds;
}

void Redline::CudaMeshData::Dispose()
{
	if (Triangles != nullptr) 
	{
		cudaChecked(cudaFree(Triangles));
		cudaChecked(cudaFree(Verticies));
		cudaChecked(cudaFree(Uv0));
		cudaChecked(cudaFree(Normals));
		cudaChecked(cudaFree(BiTangents));
		cudaChecked(cudaFree(Tangents));

		memset(this, 0, sizeof(CudaMeshData));
	}
}
