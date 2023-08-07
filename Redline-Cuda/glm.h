#pragma once

#define GLM_FORCE_CUDA 
#define GLM_FORCE_SWIZZLE 

#include <vec2.hpp>
#include <vec3.hpp>
#include <vec4.hpp>

#include <mat4x4.hpp>
#include <ext.hpp>

#include <common.hpp>
#include <exponential.hpp>
#include <geometric.hpp>
#include <matrix.hpp>

#include "gtx/quaternion.hpp"
#include <gtx/matrix_decompose.hpp>

inline void DecomposeMatrix_GLM(const glm::mat4& matrix,
	glm::vec3& outPosition,
	glm::quat& outRotation,
	glm::vec3& outScale)
{
	glm::quat tempRotation;
	glm::vec3 skew;
	glm::vec4 perspective;

	glm::decompose(matrix, outScale, tempRotation, outPosition, skew, perspective);
	outRotation = glm::conjugate(tempRotation);
}