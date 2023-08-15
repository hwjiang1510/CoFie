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
include third-party/cnpy/CMakeFiles/cnpy-static.dir/depend.make

# Include the progress variables for this target.
include third-party/cnpy/CMakeFiles/cnpy-static.dir/progress.make

# Include the compile flags for this target's objects.
include third-party/cnpy/CMakeFiles/cnpy-static.dir/flags.make

third-party/cnpy/CMakeFiles/cnpy-static.dir/cnpy.cpp.o: third-party/cnpy/CMakeFiles/cnpy-static.dir/flags.make
third-party/cnpy/CMakeFiles/cnpy-static.dir/cnpy.cpp.o: ../third-party/cnpy/cnpy.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/vision/hwjiang/3d_generation/DeepLocalShapes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object third-party/cnpy/CMakeFiles/cnpy-static.dir/cnpy.cpp.o"
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cnpy-static.dir/cnpy.cpp.o -c /vision/hwjiang/3d_generation/DeepLocalShapes/third-party/cnpy/cnpy.cpp

third-party/cnpy/CMakeFiles/cnpy-static.dir/cnpy.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cnpy-static.dir/cnpy.cpp.i"
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /vision/hwjiang/3d_generation/DeepLocalShapes/third-party/cnpy/cnpy.cpp > CMakeFiles/cnpy-static.dir/cnpy.cpp.i

third-party/cnpy/CMakeFiles/cnpy-static.dir/cnpy.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cnpy-static.dir/cnpy.cpp.s"
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /vision/hwjiang/3d_generation/DeepLocalShapes/third-party/cnpy/cnpy.cpp -o CMakeFiles/cnpy-static.dir/cnpy.cpp.s

# Object files for target cnpy-static
cnpy__static_OBJECTS = \
"CMakeFiles/cnpy-static.dir/cnpy.cpp.o"

# External object files for target cnpy-static
cnpy__static_EXTERNAL_OBJECTS =

third-party/cnpy/libcnpy.a: third-party/cnpy/CMakeFiles/cnpy-static.dir/cnpy.cpp.o
third-party/cnpy/libcnpy.a: third-party/cnpy/CMakeFiles/cnpy-static.dir/build.make
third-party/cnpy/libcnpy.a: third-party/cnpy/CMakeFiles/cnpy-static.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/vision/hwjiang/3d_generation/DeepLocalShapes/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libcnpy.a"
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy && $(CMAKE_COMMAND) -P CMakeFiles/cnpy-static.dir/cmake_clean_target.cmake
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cnpy-static.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
third-party/cnpy/CMakeFiles/cnpy-static.dir/build: third-party/cnpy/libcnpy.a

.PHONY : third-party/cnpy/CMakeFiles/cnpy-static.dir/build

third-party/cnpy/CMakeFiles/cnpy-static.dir/clean:
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy && $(CMAKE_COMMAND) -P CMakeFiles/cnpy-static.dir/cmake_clean.cmake
.PHONY : third-party/cnpy/CMakeFiles/cnpy-static.dir/clean

third-party/cnpy/CMakeFiles/cnpy-static.dir/depend:
	cd /vision/hwjiang/3d_generation/DeepLocalShapes/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /vision/hwjiang/3d_generation/DeepLocalShapes /vision/hwjiang/3d_generation/DeepLocalShapes/third-party/cnpy /vision/hwjiang/3d_generation/DeepLocalShapes/build /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy /vision/hwjiang/3d_generation/DeepLocalShapes/build/third-party/cnpy/CMakeFiles/cnpy-static.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : third-party/cnpy/CMakeFiles/cnpy-static.dir/depend

