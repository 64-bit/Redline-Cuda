#include "CompiledScene.h"
#include <vector>
#include <functional>
#include <glm.h>

#include <Scene/Scene.h>
#include <Scene/SceneObject.h>
#include <Scene/Components/Renderable/CudaMeshRenderer.h>
#include <Renderer/CudaSurface.h>
#include <Utilities/CudaUtils.h>

using namespace Redline;
using namespace std;
using namespace glm;

bool GetSurfacesFromSceneObject2(SceneObject* sceneObject,
	vector<CudaSurface>* destinationList)
{

	auto surfaceAttempt = sceneObject->TryGetComponent<CudaMeshRenderer>();

	if (surfaceAttempt == nullptr)
	{
		return true;
	}

	CudaSurface surface;

	surface.Transform = inverse(sceneObject->Transform.GetWorldTransformMatrix());
	surface.MeshId = surfaceAttempt->MeshIndex;

	destinationList->push_back(surface);

	//destinationList->push_back(surfaceAttempt->Mesh->_cudaMeshData);
	//transformList->push_back(inverse(sceneObject->Transform.GetWorldTransformMatrix()));

	return true;
}

CompiledScene::CompiledScene()
{
	SurfaceCount = 0;
	Surfaces = nullptr;
	Meshes = nullptr;
}

__host__ CompiledScene::CompiledScene(std::shared_ptr<Scene> scene)
{
	vector<CudaSurface> cudaSurfaces;

	std::function<bool(SceneObject*)> boundCallback =
		std::bind(&GetSurfacesFromSceneObject2, std::placeholders::_1, &cudaSurfaces);

	scene->ForEachSceneObject(boundCallback);

	Surfaces = CudaUtils::UploadVector(cudaSurfaces);
	SurfaceCount = cudaSurfaces.size();

	vector<CudaMesh> gpuMeshes;
	for (int i = 0; i < scene->Meshes.size(); i++)
	{
		gpuMeshes.push_back(scene->Meshes[i]->_cudaMeshData);
	}

	Meshes = CudaUtils::UploadVector(gpuMeshes);

	//TODO:Build top-level BVH here
}

__host__ void Redline::CompiledScene::Dispose()
{
	if (Surfaces != nullptr)
	{
		cudaChecked(cudaFree(Surfaces));
	}
}
