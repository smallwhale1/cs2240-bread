#include "bread.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "marching.h"
#include <omp.h>
#include <algorithm>

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QFileInfo>
#include <cmath>

using namespace std;


Bread::Bread() {
}

int Bread::toIndex(int x, int y, int z, int dimX, int dimY) {
    return x + y * dimX + z * dimX * dimY;
}

void Bread::init() {
    // specify voxel filepath

    // absolute right now
    const std::string& filepath = "meshes-binvox/bread_128.binvox";

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file");
    }

    // load in voxel into bool vector

    std::string line;
    std::getline(file, line);
    if (line != "#binvox 1") {
        throw std::runtime_error("Invalid header");
    }

    while (std::getline(file, line)) {
        if (line.substr(0, 3) == "dim") {
            sscanf(line.c_str(), "dim %d %d %d", &dimX, &dimY, &dimZ);
        } else if (line.substr(0, 10) == "translate ") {
            // handle translate
            sscanf(line.c_str(), "translate %f %f %f", &translateX, &translateY, &translateZ);
        } else if (line.substr(0, 6) == "scale ") {
            // handle scale
            sscanf(line.c_str(), "scale %f", &scale);
        } else if (line == "data") {
            // end of header
            break;
        }
    }

    // data
    // # voxels
    int numVoxels = dimX * dimY * dimZ;

    m_voxels.resize(numVoxels);
    m_P.resize(numVoxels);
    std::fill(m_P.begin(), m_P.end(), 1);
    // std::vector<bool> voxels(numVoxels);

    // parse data
    unsigned char value = 0;
    unsigned char count = 0;
    int index = 0;

    while (file.read(reinterpret_cast<char*>(&value), 1) &&
           file.read(reinterpret_cast<char*>(&count), 1)) {

        for (int i = 0; i < count; ++i) {
            m_voxels[index++] = (value != 0);
        }
        auto x = 1;
    }

    // if (index != numVoxels) {
    //     cout << "not matched" << endl;
    // }

    file.close();

    // int dim_X = 10, dim_Y = 10, dim_Z = 10;
    // vector<bool> voxels(dim_X * dim_Y * dim_Z, false);

    // // Fill a 3x3x3 cube in the center
    // for (int z = 4; z < 7; ++z)
    //     for (int y = 4; y < 7; ++y)
    //         for (int x = 4; x < 7; ++x)
    //             voxels[toIndex(x, y, z, dim_X, dim_Y)] = true;

    // extractVoxelSurfaceToOBJ(voxels, dim_X, dim_Y, dim_Z, "output.obj");

    // add padding around the edges to allow for rising
    addPadding(5);

    m_P.resize(m_voxels.size());
    std::fill(m_P.begin(), m_P.end(), 1);

    fillIn();

    // NAIVE
    // extractVoxelSurfaceToOBJ(m_voxels, dimX, dimY, dimZ, "bread-output.obj");

    // vector<Eigen::Vector3f> vertices;
    // // vector<Eigen::Vector3f> normals;
    // vector<Triangle> triangles;

    // marchingCubes(m_voxels, dimX, dimY, dimZ, vertices, triangles, edgeTable, triangleTable);

    // saveOBJ("bread_mesh_128.obj", vertices, triangles);

    // int x, y, z;
    // voxelToIndices(200, x, y, z);
    // std::cout << "x: " << x << std::endl;
    // std::cout << "y: " << y << std::endl;
    // std::cout << "z: " << z << std::endl;

    // int i;
    // indicesToVoxel(x, y, z, i);
    // std::cout << "i: " << i << std::endl;

    // BREAD LOGIC

    distanceVoxels();
    generateSphere(0, 0, 0, 2);
    generateBubbles(1, 7);

    std::vector<bool> voxelCopy = m_voxels;
    // do cross section
    for (int i = 0; i < m_voxels.size(); i++) {
        int x, y, z;
        voxelToIndices(i, x, y, z);
        // cout << "x: " << x << endl;
        // cout << "y: " << y << endl;
        // cout << "z: " << z << endl;
        if (y < dimY / 2) {
            // cout << "hi" << endl;
            // set to 0
            voxelCopy[i] = 0;
        }
    }

    writeBinvox("test-original-no-rise.binvox", dimX, dimY, dimZ, voxelCopy, translateX, translateY, translateZ, scale);

    constructMockTemp();
    generateGaussianFilter();
    convolveGaussian();

    // // std::vector<std::vector<float>> gradient = calcGradient(100);
    // // std::cout << gradient[0][5] << std::endl;
    // // std::cout << gradient[1][5] << std::endl;
    // // std::cout << gradient[2][5] << std::endl;

    m_gradVector = calcGradient(m_mock_temp);

    warpBubbles(m_gradVector);
    // rise(m_gradVector);

    for (int i = 0; i < m_voxels.size(); i++) {
        int x, y, z;
        voxelToIndices(i, x, y, z);
        // cout << "x: " << x << endl;
        // cout << "y: " << y << endl;
        // cout << "z: " << z << endl;
        if (y < dimY / 2) {
            // cout << "hi" << endl;
            // set to 0
            m_voxels[i] = 0;
        }
    }

    writeBinvox("test-no-rise.binvox", dimX, dimY, dimZ, m_voxels, translateX, translateY, translateZ, scale);

    // cout << "done!" << endl;
}

void Bread::distanceVoxels() {
    m_distance_voxels.resize(m_voxels.size());

    #pragma omp parallel for
    for (int i = 0; i < m_voxels.size(); i++) {
        if (!m_voxels[i]) {
            m_distance_voxels[i] = -1.f;
        } else {
            int x, y, z;
            voxelToIndices(i, x, y, z);
            // cout << "x: " << x << endl;
            // cout << "y: " << y << endl;
            // cout << "z: " << z << endl;
            float minDistance = dimX * dimY * dimZ;
            for (int j = 0; j < m_voxels.size(); j++) {
                if (i != j && !m_voxels[j]) {
                    int x_prime, y_prime, z_prime;
                    voxelToIndices(j, x_prime, y_prime, z_prime);
                    float currDistance = sqrt(std::pow(x - x_prime, 2) + std::pow(y - y_prime, 2) + std::pow(z - z_prime, 2));
                    if (currDistance < minDistance) {
                        minDistance = currDistance;
                    }
                }
            }
            m_distance_voxels[i] = minDistance;
        }
    }
}

void Bread::addPadding(int paddingAmt) {
    int newDimX = dimX + paddingAmt * 2;
    int newDimY = dimY + paddingAmt * 2;
    int newDimZ = dimZ + paddingAmt * 2;

    vector<bool> newVoxels(newDimX * newDimY * newDimZ, false);

    for (int x = 0; x < newDimX; x++) {
        for (int y = 0; y < newDimY; y++) {
            for (int z = 0; z < newDimZ; z++) {
                int oldIndX = x - paddingAmt;
                int oldIndY = y - paddingAmt;
                int oldIndZ = z - paddingAmt;

                int newIndex = y + z * newDimZ + x * newDimX * newDimZ;

                if (oldIndX < 0 || oldIndY < 0 || oldIndZ < 0 ||
                    oldIndX >= dimX || oldIndY >= dimY || oldIndZ >= dimZ) {
                    newVoxels[newIndex] = 0;
                } else {
                    int oldIndex;
                    indicesToVoxel(oldIndX, oldIndY, oldIndZ, oldIndex);
                    newVoxels[newIndex] = m_voxels[oldIndex];
                }
            }
        }
    }

    m_voxels = std::move(newVoxels);

    dimX = newDimX;
    dimY = newDimY;
    dimZ = newDimZ;
}

void Bread::voxelToIndices(int index, int &x, int &y, int &z) {
    // x = index % (dimX * dimZ);
    // index -= x * dimX * dimZ;
    // z = index % dimZ;
    // y = index - z * dimZ;

    int a = dimX * dimY;
    z = index / a;

    int b = index - a * z;
    y = b / dimX;
    x = b % dimX;
}

void Bread::indicesToVoxel(int x, int y, int z, int &index) {
    index = y;
    index += z * dimZ;
    index += x * dimX * dimZ;
}

void Bread::voxelToSpatialCoords(int x, int y, int z, float &worldX, float &worldY, float &worldZ) {
    worldX = (x + 0.5f) / dimX;
    worldY = (y + 0.5f) / dimY;
    worldZ = (z + 0.5f) / dimZ;
    worldX = scale * worldX + translateX;
    worldY = scale * worldY + translateY;
    worldZ = scale * worldZ + translateZ;
}

bool Bread::voxelAt(int x, int y, int z) {
    return m_voxels[x * dimX * dimZ + z * dimZ + y];
}

void Bread::generateSphere(int x, int y, int z, int radius) {
    // get bounding box of size 2R + 1 in each dimension and store in vector
    // iterate through vector and check distance; if distance is < radius, set voxel == 0
    std::vector<float> boundingBox;
    int boundingBoxWidth = 2 * radius + 1;
    boundingBox.assign(pow(boundingBoxWidth, 3), 0);

    int count = 0;

    for (int i = -1 * radius; i <= radius; i++) {
        for (int j = -1 * radius; j <= radius; j++) {
            for (int k = -1 * radius; k <= radius; k++) {
                int idx;
                int x_prime = x + i;
                int y_prime = y + j;
                int z_prime = z + k;

                // bounds checking. TODO check: >= or > dimD depends on how dims are parsed. TODO: verify
                if (x_prime < 0 || x_prime >= dimX || y_prime < 0 || y_prime >= dimY || z_prime < 0 || z_prime >= dimZ) {
                    continue;
                }

                indicesToVoxel(x_prime, y_prime, z_prime, idx);

                // sanity bounds check. TODO: verify
                if (idx < 0 || idx > m_voxels.size()) {
                    continue;
                }

                float distance = sqrt(std::pow(x - x_prime, 2) + std::pow(y - y_prime, 2) + std::pow(z - z_prime, 2));
                // m_voxels[idx] = 1;

                // std::cout << distance << std::endl; // uncomment for checking that distance calculations are correct
                // add some padding around the crust
                if (distance <= radius && m_distance_voxels[idx] >= 3) {
                    m_voxels[idx] = 0;
                    // modify P
                    m_P[idx] = std::max(m_P[idx], radius);
                    count++;
                } else {
                    // m_voxels[idx] = 1; // TODO: prob can get rid of this since we want to maintain original mesh binary status; this is just for initial checking.
                }
            }
        }
    }

    // std::cout << "count: " << count << std::endl; // uncomment to check that we're subtracting the correct number of voxels for a given sphere
}

// use min of 2 and max of 20 for baguette setting
void Bread::generateBubbles(int minRadius, int maxRadius) {
    int radius = minRadius;

    // see page 9 for some constants. currently using baguette settings
    int r = 144; // resolution of proving vol in each spatial coordinate
    float k = 0.07 * pow(r, 3) * 0.05; // the amount of actual spheres at each radius
    float d = 2.78; // fractal exponent for likelihood of spheres given radii
    while (radius <= maxRadius) {
        // subtract spheres of minRadius
        int numSpheres = k / pow(radius, d);
        std::cout << "numSpheres: " << numSpheres << std::endl;
        for (int i = 0; i < numSpheres; i++) {
            // generate random coord inside dims
            int x = arc4random_uniform(dimX);
            int y = arc4random_uniform(dimY);
            int z = arc4random_uniform(dimZ);
            generateSphere(x, y, z, radius);
        }
        // increment
        radius++;
    }
}

void Bread::fillIn() {
    for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {

            int state = 0;
            int startZ = -1;
            for (int z = 0; z < dimZ; z++) {
                if (state == 0) {
                    if (voxelAt(x, y, z)) {
                        state = 1;
                        startZ = z;
                    }
                } else {
                    if (voxelAt(x, y, z)) {
                        state = 0;
                        if (startZ + 1 < z) {
                            for (int w = startZ + 1; w < z; w++) {
                                int index;
                                indicesToVoxel(x, y, w, index);
                                bool zz = m_voxels[index];
                                if (m_voxels[index] == 0) {
                                   // cout << "filling in" << endl;
                                }
                                m_voxels[index] = 1;

                            }
                        }
                    }
                    // else {
                    //     int index;
                    //     indicesToVoxel(x, y, z, index);
                    //     m_voxels[index] = 1;
                    // }
                }
            }
        }
    }
}
