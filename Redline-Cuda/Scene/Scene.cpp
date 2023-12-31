#include "Scene.h"
#include "SceneObject.h"

using namespace Redline;
using namespace std;

Scene::Scene(const string& sceneName)
{
	Name = sceneName;

	ScenegraphRoot = make_shared<SceneObject>(*this, "[Scene Root]");
	LivingSceneObjects.push_back(ScenegraphRoot);
	EnvironmentPower = 1.0f;
	BackgroundColor = mathfu::vec3(0.4f, 0.4f, 0.4f);
}

std::shared_ptr<SceneObject> Scene::CreateSceneObject(const string& name)
{
	//Create object
	auto newSceneObject = std::make_shared<SceneObject>(*this, name);

	//Set parent as the scene root
	newSceneObject->Transform.SetParent(&ScenegraphRoot->Transform);

	//Add to list of living objects
	LivingSceneObjects.push_back(newSceneObject);

	return newSceneObject;
}

void Scene::ForEachSceneObject(std::function<bool(SceneObject*)>& callback)
{
	ForEachSceneObject(callback, ScenegraphRoot.get());
}

SceneObject* Scene::FindObjectByName(const string& name)
{
	SceneObject* result = nullptr;
	function<bool(SceneObject*)> callback = [&result, &name](SceneObject* obj)
	{
		//Correction_Camera
		if (obj->Name == name)
		{
			result = obj;
			return false;
		}

		return true;
	};
	ForEachSceneObject(callback);
	return result;
}

int Redline::Scene::GetIndexForImage(std::shared_ptr<Bitmap2D> image)
{
	for (int i = 0; i < Images.size(); i++) 
	{
		if (Images[i] == image) 
		{
			return i;
		}
	}
	return -1;
}

int Redline::Scene::GetIndexForMaterial(std::shared_ptr<Material> material)
{
	for (int i = 0; i < Images.size(); i++)
	{
		if (Materials[i] == material)
		{
			return i;
		}
	}
	return -1;
}

int Redline::Scene::GetIndexForMesh(std::shared_ptr<CudaMeshBuilder> mesh)
{
	for (int i = 0; i < Images.size(); i++)
	{
		if (Meshes[i] == mesh)
		{
			return i;
		}
	}
	return -1;
}

void Scene::ForEachSceneObject(std::function<bool(SceneObject*)>& callback, SceneObject* target)
{
	const bool decend = callback(target);

	if(!decend)
	{
		return;
	}

	const auto children = target->Transform.GetChildren();
	for(int i = 0; i < children.size(); i++)
	{
		auto child = children[i];
		ForEachSceneObject(callback, &child->Object);
	}
}
