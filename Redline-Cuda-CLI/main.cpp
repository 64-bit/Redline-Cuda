#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <SDL.h>
#include <string>
#include <tchar.h>

#include "RenderApplication.h"
#include "CommandLineArguments.h"
#include "CLIErrorCodes.h"

#undef main

using namespace Redline;
int main_asdf();

int main(int argc, char** argv)
{
	RenderApplication application;
	CommandLineArguments* commandLineArguments;

	const int argsErrorCode = CommandLineArguments::ParseCommandLineArguments(argc, argv, commandLineArguments);

	commandLineArguments->SceneFilename = "TestData/Root.gltf";

	if (argsErrorCode != CLI_ERRORCODE___OK)
	{
		if (argsErrorCode == CLI_ERRORCODE___HELP_PRINTED)
		{
			return CLI_ERRORCODE___OK;	//This special case causes a normal return if the help message was printed,
			//but prevents the applicatino from running
		}
		return argsErrorCode;//The command line parser will print a detailed error message, just return the code it provided
	}

	const auto applicationResult = application.Run(commandLineArguments);
	delete commandLineArguments;
	return applicationResult;
}

