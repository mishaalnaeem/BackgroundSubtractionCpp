## Setting Up C++ Envrionment

Following are required to run this code:

* Gcc
* CMake
* Libtorch
* OpenCV

Install CMake:

Based on OS, install the latest CMake https://cmake.org

Setting up Libtorch:

1. Download libtorch zip from here https://pytorch.org
2. For MacOS, Unzip libtorch, and navigate to "TorchConfig.cmake" file
replace the line containing "find_library" command for c10 with
    ```find_library(C10_LIBRARY c10 NO_CMAKE_FIND_ROOT_PATH PATH_SUFFIXES .dylib  PATHS "${TORCH_INSTALL_PREFIX}/lib")```

Setting up OpenCV:

1. Follow OS based instructions on https://docs.opencv.org/4.x/df/d65/tutorial_table_of_content_introduction.html
2. If CMake is not detecting OpenCV, Find the path for opencv2 after installtion and add:
set(OpenCV_DIR "/path/to/include/opencv4/opencv2")

### Running the project

1. Open terminal in project directory, and create build directory:

```mkdir build```

2. Navigate into build directory:

```cd build```

3. Configure the path to libtorch:

```cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..```

4. Build the project:

```cmake --build . --config Release```

5. Run the project:
```./BackgroundSubtraction```


### About the project
Model name: output_model.pt

Model is v2net. Used pretrained: https://www.kaggle.com/code/remekkinas/remove-background-salient-object-detection/notebook 

Conversion script: https://colab.research.google.com/drive/172T9E7XftxeXkg-qVe6yOxfkHy-60K3V?usp=sharing 

