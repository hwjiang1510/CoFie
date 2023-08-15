project("DeepSDF")
cmake_minimum_required(VERSION 3.0)

include_directories(/vision/hwjiang/repos/CLI11/include)
include_directories(/vision/hwjiang/repos/Pangolin/include)
include_directories(/vision/hwjiang/repos/Pangolin/build/src/include)
include_directories(/vision/hwjiang/repos/nanoflann/include)
include_directories(/vision/hwjiang/repos/eigen3/eigen-3.4.0)

set(Pangolin_DIR /vision/hwjiang/repos/Pangolin/build/src)
set(nanoflann_DIR /vision/hwjiang/repos/nanoflann/include)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)

#find_package(CLI11 CONFIG REQUIRED)
#find_package(Eigen3 REQUIRED)
#find_package(nanoflann REQUIRED)
#find_package(Pangolin REQUIRED)

add_subdirectory(third-party/cnpy)

set(SOURCES
    src/ShaderProgram.cpp
    src/Utils.cpp)

add_executable(PreprocessMesh src/PreprocessMesh.cpp ${SOURCES})
target_include_directories(PreprocessMesh PRIVATE third-party/cnpy)
target_link_libraries(PreprocessMesh PRIVATE ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${Pangolin_DIR}/libpangolin.so cnpy)
target_compile_features(PreprocessMesh PRIVATE cxx_std_14)
set_target_properties(PreprocessMesh PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")

add_executable(SampleVisibleMeshSurface src/SampleVisibleMeshSurface.cpp ${SOURCES})
target_include_directories(SampleVisibleMeshSurface PRIVATE third-party/cnpy)
target_link_libraries(SampleVisibleMeshSurface PRIVATE ${GLEW_LIBRARIES} ${OPENGL_gl_LIBRARY} ${CMAKE_THREAD_LIBS_INIT} ${Pangolin_DIR}/libpangolin.so cnpy)
target_compile_features(SampleVisibleMeshSurface PRIVATE cxx_std_14)
set_target_properties(SampleVisibleMeshSurface PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${PROJECT_SOURCE_DIR}/bin")
