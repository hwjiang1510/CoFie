# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /vision/hwjiang/3d_generation/DeepLocalShapes

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /vision/hwjiang/3d_generation/DeepLocalShapes/build

# Include any dependencies generated for this target.
include CMakeFiles/PreprocessMesh.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PreprocessMesh.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PreprocessMesh.dir/flags.make

CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.o: CMakeFiles/PreprocessMesh.dir/flags.make
CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.o: ../src/PreprocessMesh.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/vision/hwjiang/3d_generation/DeepLocalShapes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.o -c /vision/hwjiang/3d_generation/DeepLocalShapes/src/PreprocessMesh.cpp

CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /vision/hwjiang/3d_generation/DeepLocalShapes/src/PreprocessMesh.cpp > CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.i

CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /vision/hwjiang/3d_generation/DeepLocalShapes/src/PreprocessMesh.cpp -o CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.s

CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.o: CMakeFiles/PreprocessMesh.dir/flags.make
CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.o: ../src/ShaderProgram.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/vision/hwjiang/3d_generation/DeepLocalShapes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.o -c /vision/hwjiang/3d_generation/DeepLocalShapes/src/ShaderProgram.cpp

CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /vision/hwjiang/3d_generation/DeepLocalShapes/src/ShaderProgram.cpp > CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.i

CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /vision/hwjiang/3d_generation/DeepLocalShapes/src/ShaderProgram.cpp -o CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.s

CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.o: CMakeFiles/PreprocessMesh.dir/flags.make
CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.o: ../src/Utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/vision/hwjiang/3d_generation/DeepLocalShapes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.o -c /vision/hwjiang/3d_generation/DeepLocalShapes/src/Utils.cpp

CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /vision/hwjiang/3d_generation/DeepLocalShapes/src/Utils.cpp > CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.i

CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /vision/hwjiang/3d_generation/DeepLocalShapes/src/Utils.cpp -o CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.s

# Object files for target PreprocessMesh
PreprocessMesh_OBJECTS = \
"CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.o" \
"CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.o" \
"CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.o"

# External object files for target PreprocessMesh
PreprocessMesh_EXTERNAL_OBJECTS =

../bin/PreprocessMesh: CMakeFiles/PreprocessMesh.dir/src/PreprocessMesh.cpp.o
../bin/PreprocessMesh: CMakeFiles/PreprocessMesh.dir/src/ShaderProgram.cpp.o
../bin/PreprocessMesh: CMakeFiles/PreprocessMesh.dir/src/Utils.cpp.o
../bin/PreprocessMesh: CMakeFiles/PreprocessMesh.dir/build.make
../bin/PreprocessMesh: /usr/lib/x86_64-linux-gnu/libGLEW.so
../bin/PreprocessMesh: /usr/lib/x86_64-linux-gnu/libGL.so
../bin/PreprocessMesh: /vision/hwjiang/repos/Pangolin/build/src/libpangolin.so
../bin/PreprocessMesh: third-party/cnpy/libcnpy.so
../bin/PreprocessMesh: /usr/lib/x86_64-linux-gnu/libz.so
../bin/PreprocessMesh: CMakeFiles/PreprocessMesh.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/vision/hwjiang/3d_generation/DeepLocalShapes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable ../bin/PreprocessMesh"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PreprocessMesh.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PreprocessMesh.dir/build: ../bin/PreprocessMesh

.PHONY : CMakeFiles/PreprocessMesh.dir/build

CMakeFiles/PreprocessMesh.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PreprocessMesh.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PreprocessMesh.dir/clean

CMakeFiles/PreprocessMesh.dir/depend:
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /vision/hwjiang/3d_generation/DeepLocalShapes /vision/hwjiang/3d_generation/DeepLocalShapes /vision/hwjiang/3d_generation/DeepLocalShapes/build /vision/hwjiang/3d_generation/DeepLocalShapes/build /vision/hwjiang/3d_generation/DeepLocalShapes/build/CMakeFiles/PreprocessMesh.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PreprocessMesh.dir/depend

