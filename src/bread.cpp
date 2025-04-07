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

    int dimX;
    int dimY;
    int dimZ;

    while (std::getline(file, line)) {
        if (line.substr(0, 3) == "dim") {
            sscanf(line.c_str(), "dim %d %d %d", &dimX, &dimY, &dimZ);
        } else if (line.substr(0, 10) == "translate ") {
            // handle translate
        } else if (line.substr(0, 6) == "scale ") {
            // handle scale
        } else if (line == "data") {
            // end of header
            break;
        }
    }

    // data
    // # voxels
    int numVoxels = dimX * dimY * dimZ;

    cout << "Num voxels: " << numVoxels << endl;

    std::vector<bool> voxels(numVoxels);

    // populate voxels
    char val;
    int index = 0;
    while (file.read(reinterpret_cast<char*>(&val), 1)) {
        for (int i = 0; i < 8; ++i) {
            if (index < numVoxels) {
                voxels[index++] = (val >> i) & 1;
            }
        }
    }

    // cout << voxels[0] << endl;

    file.close();

}
