#include "bread.h"
#include <iostream>
#include <fstream>

void Bread::writeBinvox(const std::string& filename, int dimX, int dimY, int dimZ, const std::vector<bool>& voxels, float translateX, float translateY, float translateZ, float scale) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) {
        throw std::runtime_error("error opening output file");
    }

    // header
    output << "#binvox 1\n";
    output << "dim " << dimX << " " << dimY << " " << dimZ << "\n";
    output << "translate " << translateX << " " << translateY << " " << translateZ <<"\n";
    output << "scale " << scale << "\n";
    output << "data\n";

    unsigned char currVal = voxels[0] ? 1 : 0;
    unsigned char count = 0;

    for (bool v : voxels) {
        unsigned char voxel = v ? 1 : 0;
        if (voxel == currVal) {
            ++count;
            if (count == 255) {
                output.put(currVal);
                output.put(count);
                count = 0;
            }
        } else {
            if (count > 0) {
                output.put(currVal);
                output.put(count);
            }
            currVal = voxel;
            count = 1;
        }
    }

    if (count > 0) {
        output.put(currVal);
        output.put(count);
    }

    output.close();
}
