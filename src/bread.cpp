#include "bread.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QFileInfo>
#include <cmath>

using namespace std;

Bread::Bread() {
}

void Bread::init() {
    // specify voxel filepath

    // absolute right now
    const std::string& filepath = "meshes-binvox/bun_48x23x48.binvox";

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
    // std::vector<bool> voxels(numVoxels);

    // populate voxels
    char val;
    int index = 0;
    while (file.read(reinterpret_cast<char*>(&val), 1)) {
        for (int i = 0; i < 8; ++i) {
            if (index < numVoxels) {
                m_voxels[index++] = (val >> i) & 1;
            }
        }
    }

    // cout << voxels[0] << endl;
    cout << "Num voxels: " << numVoxels << endl;

    file.close();

    distanceVoxels();

    cout << "done!" << endl;
}

void Bread::distanceVoxels() {
    m_distance_voxels.resize(m_voxels.size());
    for (int i = 0; i < m_voxels.size(); i++) {
        if (!m_voxels[i]) {
            m_distance_voxels[i] = 0.f;
        } else {
            int x, y, z;
            voxelToIndices(i, x, y, z);
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

void Bread::voxelToIndices(int index, int &x, int &y, int &z) {
    x = index % (dimX * dimZ);
    index -= x * dimX * dimZ;
    z = index % dimZ;
    y = index - z * dimZ;
}

void Bread::voxelToSpatialCoords(int x, int y, int z, float &worldX, float &worldY, float &worldZ) {
    worldX = (x - 0.5f) / dimX;
    worldY = (y + 0.5f) / dimY;
    worldZ = (z + 0.5f) / dimZ;
    worldX = scale * worldX + translateX;
    worldY = scale * worldY + translateY;
    worldZ = scale * worldZ + translateZ;
}

bool Bread::voxelAt(int x, int y, int z) {
    return m_voxels[x * dimX * dimZ + z * dimZ + y];
}
