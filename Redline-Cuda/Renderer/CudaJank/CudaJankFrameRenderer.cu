#include "CudaJankFrameRenderer.h"
#include <GraphicalResources/Bitmap2D.h>
#include <functional>
#include <Scene/SceneObject.h>
#include <Scene/Scene.h>
#include <Utilities/CudaUtils.h>
#include <Scene/Components/Renderable/CudaMeshRenderer.h>
#include <mathfu/glsl_mappings.h>
#include <Math/Ray.h>
#include <Scene/CompiledScene.h>
#include <Renderer/CudaSurface.h>
//#include <Math/cuda_helper_math.h>
//#include <Math/cudaMat4.h>

using namespace Redline;
using namespace std;
using namespace glm;

struct PackedCameraData
{
	vec3 cameraPosition;
	quat cameraRotation;
	float fovX;
	float fovY;
};


__device__ __host__ inline bool RayTriangleIntersection(const Ray& localRay, const vec3& vertA, const vec3& vertB,
	const vec3& vertC, float& outHitDistance)
{
	const float EPSILON = 0.0000001f;

	vec3 edge1 = vertB - vertA;
	vec3 edge2 = vertC - vertA;

	vec3 h = cross(localRay.Direction, edge2);
	float a = dot(edge1, h);


	if (a > -EPSILON && a < EPSILON)
	{
		return false;
	}

	/*if (a < EPSILON) Replace the above with this for backface culling
	{
		return false;
	}*/

	float f = 1.0f / a;
	vec3 s = localRay.Origin - vertA;
	float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f)
	{
		return false;
	}

	vec3 q = cross(s, edge1);
	float v = f * dot(localRay.Direction, q);
	if (v < 0.0f || u + v > 1.0f)
	{
		return false;
	}

	float t = f * dot(edge2, q);

	if (t > EPSILON) // ray intersection
	{
		outHitDistance = t;
		return true;
	}
	else // This means that there is a line intersection but not a ray intersection.
	{
		return false;
	}
}

__global__ void JankRenderFrame(CudaBitmapData frameBuffer, PackedCameraData camera, CompiledScene compiledScene)
{
	uint2 i;
	i.x = threadIdx.x + blockIdx.x * blockDim.x;
	i.y = threadIdx.y + blockIdx.y * blockDim.y;

	int w = i.x;
	int h = i.y;

	float xSSPosition = ((float)w) / (float)frameBuffer.Width;
	xSSPosition = xSSPosition;
	xSSPosition = (xSSPosition * 2.0f) - 1.0f;

	float ySSPosition = ((float)h) / (float)frameBuffer.Height;
	ySSPosition = 1.0f - ySSPosition;
	ySSPosition = (ySSPosition * 2.0f) - 1.0f;

	float xDir = xSSPosition * tanf(camera.fovX * 0.5f);
	float yDir = ySSPosition * tanf(camera.fovY * 0.5f);

	Ray ray;
	ray.Origin = camera.cameraPosition;

	ray.Direction = normalize(vec3(xDir, yDir, -1.0f));
	ray.Direction = camera.cameraRotation * ray.Direction;

	Color color;
	color.A = 0;
	color.R = 0;
	color.G = 0;
	color.B = 0;

	float bestHitDistance;

	for (int surfaceIndex = 0; surfaceIndex < compiledScene.SurfaceCount; surfaceIndex++)
	{
		CudaSurface surface = compiledScene.Surfaces[surfaceIndex];
		mat4 worldToLocal = surface.Transform;
		CudaMesh mesh = compiledScene.Meshes[surface.MeshId];

		Ray localRay;
		localRay.Origin = (worldToLocal * vec4(ray.Origin, 1.0f)).xyz;
		localRay.Direction = (worldToLocal * vec4(ray.Direction, 0.0f)).xyz;

		float entry, exit;
		if (mesh.Bounds.DoesRayIntersect(localRay, entry, exit))
		{
			vec3* meshVerts = ((vec3*)mesh.Verticies);
			for (int tri = 0; tri < mesh.TriangleCount; tri++)
			{
				uint4 triangle = mesh.Triangles[tri];

				vec3 vertA = meshVerts[triangle.x];
				vec3 vertB = meshVerts[triangle.y];
				vec3 vertC = meshVerts[triangle.z];

				float thisHitDistance;
				bool didHit = RayTriangleIntersection(localRay, vertA, vertB, vertC, thisHitDistance);
				if (didHit) 
				{
					if (surfaceIndex == 0)
					{
						color.R = 255;
					}
					if (surfaceIndex == 1)
					{
						color.G = 255;
					}
					if (surfaceIndex == 2)
					{
						color.B = 255;
					}
					break;
				}
			}


		}
	}

	frameBuffer.SafeWriteColor(i, color);
}

CudaJankFrameRenderer::CudaJankFrameRenderer(
	std::shared_ptr<Redline::Scene>& scene,
	std::shared_ptr<CameraComponent>& camera,
	const FrameOutputSettings& frameOutputSettings,
	const FrameQuailtySettings& frameQuailtySettings)
{
	_scene = scene;
	_camera = camera;
	_frameOutputSettings = frameOutputSettings;
	_frameQuailtySettings = frameQuailtySettings;

	_renderTarget = std::make_shared<Bitmap2D>(frameOutputSettings.OutputWidth, frameOutputSettings.OutputHeight);
	_cudaRenderTarget = std::make_shared<CudaBitmap2D>(frameOutputSettings.OutputWidth, frameOutputSettings.OutputHeight);

	_compiledScene = CompiledScene(_scene);
}

bool GetSurfacesFromSceneObject(SceneObject* sceneObject,
	std::vector<CudaMesh>* destinationList,
	std::vector<mat4>* transformList)
{

	auto surfaceAttempt = sceneObject->TryGetComponent<CudaMeshRenderer>();

	if (surfaceAttempt == nullptr)
	{
		return true;
	}

	destinationList->push_back(surfaceAttempt->Mesh->_cudaMeshData);
	transformList->push_back(inverse(sceneObject->Transform.GetWorldTransformMatrix()));

	return true;
}

CudaJankFrameRenderer::~CudaJankFrameRenderer()
{
	_compiledScene.Dispose();
}

std::shared_ptr<CudaBitmap2D> CudaJankFrameRenderer::GetCurrentFrameState()
{
	return _cudaRenderTarget;
}

void CudaJankFrameRenderer::ResetRenderer()
{
	//throw std::exception();
}

void CudaJankFrameRenderer::RenderFrame()
{
	const unsigned int width = _frameOutputSettings.OutputWidth;
	const unsigned int height = _frameOutputSettings.OutputHeight;

	Color clearColor;
	clearColor.R = 255;
	clearColor.G = 128;
	clearColor.B = 0;
	clearColor.A = 255;

	_cudaRenderTarget->CudaClear(clearColor);

	PackedCameraData cameraArgs;
	cameraArgs.cameraPosition = _camera->Object.Transform.GetPosition();
	cameraArgs.cameraRotation = _camera->Object.Transform.GetRotation();
	cameraArgs.fovY = _camera->YAxisFieldofViewRadians;
	cameraArgs.fovX = cameraArgs.fovY * static_cast<float>(_frameOutputSettings.OutputWidth) / static_cast<float>(_frameOutputSettings.OutputHeight);

	dim3 block(8, 8);
	int gridx = (width / 8) + ((width % 8) > 0 ? 1 : 0);
	int gridy = (height / 8) + ((height % 8) > 0 ? 1 : 0);

	dim3 grid(gridx, gridy);

	JankRenderFrame<<<grid, block >>>(_cudaRenderTarget->Data, cameraArgs, _compiledScene);
	cudaChecked(cudaGetLastError());
	cudaChecked(cudaDeviceSynchronize());
	cudaChecked(cudaGetLastError());
}