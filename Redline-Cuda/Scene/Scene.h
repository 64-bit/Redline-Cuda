#pragma once

#include "SceneForwardDeclarations.h"

#include <memory>
#include <string>
#include "mathfu/glsl_mappings.h"
#include <vector>
#include <functional>
#include "../GraphicalResources/CubemapTexture.h"
#include "SceneObject.h"
#include <GraphicalResources/Cuda/CudaMesh.h>

namespace Redline
{

	class Material;


	/*
	 *	A Scene contains a scenegraph of SceneObjects, Environmental settings and other associated metadata
	 *
	 */
	class Scene
	{
	public:

		std::string Name;

		std::shared_ptr<SceneObject> ScenegraphRoot;

	private:
		std::vector<std::shared_ptr<SceneObject>> LivingSceneObjects;

		void ForEachSceneObject(std::function<bool(SceneObject*)>& callback, SceneObject* target);

	public:

		/*Basic scene constructor, */
		Scene(const std::string& sceneName);

		std::shared_ptr<SceneObject> CreateSceneObject(const std::string& name);

		void ForEachSceneObject(std::function<bool(SceneObject*)>& callback);

		SceneObject* FindObjectByName(const std::string& name);

		template<class T>
		std::shared_ptr<T> FindComponentByType();


		//Temp

		std::vector<std::shared_ptr<Bitmap2D>> Images;
		std::vector<std::shared_ptr<Material>> Materials;
		std::vector<std::shared_ptr<CudaMeshBuilder>> Meshes;

		int GetIndexForImage(std::shared_ptr<Bitmap2D> image);
		int GetIndexForMaterial(std::shared_ptr<Material> material);
		int GetIndexForMesh(std::shared_ptr<CudaMeshBuilder> mesh);


		mathfu::vec3 BackgroundColor;
		float EnvironmentPower;
		std::shared_ptr<CubemapTexture> EnvironmentCubemap;

		mathfu::vec3 AmbientLight;
	};


	template <class T>
	std::shared_ptr<T> Scene::FindComponentByType()
	{
		std::shared_ptr<T> result = nullptr;
		std::function<bool(SceneObject*)> callback = [&result](SceneObject* obj)
		{
			auto component = obj->TryGetComponent<T>();
			if (component != nullptr)
			{
				result = component;
				return false;
			}
			return true;
		};
		ForEachSceneObject(callback);
		return result;
	}

}
