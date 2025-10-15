#!/bin/bash

# === Load modules with Spack ===

echo "=== Loading required modules with Spack ==="

# GCC (9.3)
echo "Loading GCC..."
spack load gcc/lhqcen5

# CMake
echo "Loading CMake..."
spack load cmake/pbddesj

# OpenMPI
echo "Loading OpenMPI..."
spack load openmpi/xcunp5q

# libtiff
echo "Loading libtiff..."
spack load libtiff/gnxev37

# === Build EXAMM ===

echo "=== Building EXAMM ==="

# Create build directory if it doesn't exist
echo "Cleaning previous build directory (if exists)..."
rm -rf build

echo "Creating new build directory..."
mkdir build
cd build

# Run cmake
echo "Running cmake..."
cmake ..

# Run make
echo "Running make (this may take a few minutes)..."
make

echo "=== Build complete! ==="
