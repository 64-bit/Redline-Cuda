#pragma once
#include<memory>
#include <SDL.h>
#undef main

namespace Redline
{
	class CommandLineArguments;
	class CudaBitmap2D;

	class RenderPreviewWindow
	{
	public:

		RenderPreviewWindow();

		void SetTitle(const char* title);

		int CreateWindow(const CommandLineArguments* const arguments);

		void UpdateWindowFromBitmap(std::shared_ptr<CudaBitmap2D> bitmapToDisplay);

		bool ShouldQuitThisFrame();

	private:

		SDL_Window* _sdlWindow = nullptr;
		SDL_Surface* _sdlWindowSurface;
	};
}
