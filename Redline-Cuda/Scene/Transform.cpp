#include <glm.h>
#include "SceneObject.h"
#include "Transform.h"
#include "Scene.h"


using namespace Redline;
using namespace glm;

Transform::Transform(SceneObject& owningObject)
	: SceneObjectComponent(owningObject)
{
	_localPosition = vec3(0.0f, 0.0f, 0.0f);
	_localRotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
	_localScale = vec3(1.0f, 1.0f, 1.0f);
	Parent = nullptr;
}

mat4 Transform::GetWorldTransformMatrix() const
{
	//If we have no parent, return local transform as world matrix transform
	if(Parent == nullptr)
	{
		return GetLocalTransformMatrix();
	}

	//We do have a parent, combine our local transform with our parents transform
	return Parent->GetWorldTransformMatrix() * GetLocalTransformMatrix();
}

mat4 Transform::GetLocalTransformMatrix() const
{
	//auto rotationMatrix = _localRotation.ToMatrix();

	mat4 transform = mat4(1.0f);

	transform = glm::translate(transform, _localPosition);

	auto rotationMatrix = glm::toMat4(_localRotation);
	transform = transform * rotationMatrix;

	transform = glm::scale(transform, _localScale);

	return transform;
	//return mat4::Transform(_localPosition, rotationMatrix, _localScale);
}

void Transform::SetParent(Transform* newParent)
{
	if(newParent == Parent)
	{//Nothing has changed, do nothing
		return;
	}

	//If we had a old parent, remove ourselfs from the list of it's children
	if(Parent != nullptr)
	{
		const auto itr = std::find(Parent->Children.begin(), Parent->Children.end(), this);
		if(itr != Parent->Children.end())
		{
			Parent->Children.erase(itr);
		}
		else
		{
			//TODO: Handle this error, Error in bookeeping of Parent-Child relationship in transform, this indicates a bug and must be fixed
			assert(false);		
		}
	}

	if(newParent == nullptr)
	{
		//We are detaching from our curent object, but this would leave us floating in space, so attach
		//ourselfs to the scene root instead
		newParent = &HomeScene.ScenegraphRoot->Transform;
	}

	//If the new parent is not null, ensure it is in the correct scene, and then add ourselfs to it's children
	if(newParent != nullptr)
	{
		assert(&newParent->HomeScene == &HomeScene);
		if(&newParent->HomeScene != &HomeScene)
		{
			//TODO:Handle error
			return;
		}
		newParent->Children.push_back(this);
	}
	else
	{
		//This should not happen, due to the above change to the newParent pointer.
		assert(false);
	}

	//Set our parent to the new parent
	Parent = newParent;
}

const std::vector<Transform*>& Transform::GetChildren() const
{
	return Children;
}

vec3 Transform::GetLocalPosition() const
{
	return _localPosition;
}

vec3 Transform::GetPosition() const
{
	if(Parent == nullptr)
	{
		return _localPosition;
	}

	const auto parentPosition = Parent->GetPosition();
	const auto parentRotation = Parent->GetRotation();

	return parentPosition + parentRotation * _localPosition;
}

void Transform::SetLocalPosition(const vec3& position)
{
	_localPosition = position;
}

void Transform::SetPosition(const vec3& position)
{
	if(Parent == nullptr)
	{
		_localPosition = position;
		return;
	}

	const auto parentPosition = Parent->GetPosition();
	const auto parentRotation = Parent->GetRotation();

	_localPosition = inverse(parentRotation) * (position - parentPosition);
}

quat Transform::GetLocalRotation() const
{
	return _localRotation;
}

quat Transform::GetRotation() const
{
	if(Parent == nullptr)
	{
		return _localRotation;
	}

	return glm::toQuat(GetWorldTransformMatrix());	
	//return quat::FromMatrix(GetWorldTransformMatrix());
}

void Transform::SetLocalRotation(const quat& rotation)
{
	_localRotation = rotation;
	_localRotation = normalize(_localRotation);
}

void Transform::SetRotation(const quat& rotation)
{
	if (Parent == nullptr)
	{
		_localRotation = rotation;
		return;
	}

	auto parentRotation = Parent->GetRotation();

	//compute rotation from parent to new destination rotation
	_localRotation = inverse(parentRotation) * rotation;
	_localRotation = normalize(_localRotation);
}

vec3 Transform::GetLocalScale() const
{
	return _localScale;
}

vec3 Transform::GetScale() const
{
	//TODO: Need to decompose entire transform
	return _localScale;
}

void Transform::SetLocalScale(const vec3& scale)
{
	_localScale = scale;
}

void Transform::SetScale(const vec3& scale)
{
	//TODO: Need to decomponse entire transform
}

vec3 Transform::Forwards() const
{
	return GetRotation() * vec4(0.0f, 0.0f, -1.0f, 0.0f).xyz();
}

vec3 Transform::Right() const
{
	return GetRotation() * vec4(1.0f, 0.0f, 0.0f, 0.0f).xyz();
}

vec3 Transform::Up() const
{
	return GetRotation() * vec4(0.0f, 1.0f, 0.0f, 0.0f).xyz();
}

