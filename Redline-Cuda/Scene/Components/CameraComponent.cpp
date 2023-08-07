#include "CameraComponent.h"
#include "../SceneObject.h"
#include "../../Math/Math.h"
#include <glm.h>

using namespace Redline;
using namespace glm;

CameraComponent::CameraComponent(SceneObject& parent)
	:SceneObjectComponent(parent)
{
	YAxisFieldofViewRadians = 90.0f;
	AspectRatio = 4.0f / 3.0f;
}

mat4 CameraComponent::GetViewMatrix()
{
	auto worldMatrix = Object.Transform.GetWorldTransformMatrix();
	return inverse(worldMatrix);
}