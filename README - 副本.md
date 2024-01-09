# User Tutorials
## Run Application
Run `cmake-build-debug/opencv` (MacOS) or `cmake-build-debug/opencv.exe` (Windows) in the project directory.
## Compile Code
Make sure that you have downloaded and installed `g++`、`gcc`、`opencv`、`cmake`, and configure the environment variables accordingly. If run directly in Visual Studio, it may cause errors.
1. First cd to the project directory, for example:
```bash
cd /Users/Bureaux/Documents/workspace/CLionProjects/opencv
```
2. Compile with Cmake, **which is already configured for environment variables**, such as:
```bash
cmake --build cmake-build-debug --target opencv -- -j 6
```
3. The step 2 will generate the executable file in `cmake-build-debug`, which can be directly run.
## User Tutorials
Enter `1` to use the sample image with target sign in this report. Enter `2` is to use the sample image that comes with the project without the target sign. Enter `3` to enter the absolute path of the third-party image.
