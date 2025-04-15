#include "bread.h"
#include <iostream>
#include <algorithm>

using namespace Eigen;

void Bread::constructMockGradient() {
    // std::vector<float> mockGradients;
    m_mock_gradient.resize(m_distance_voxels.size());

    float maxTemp = 100.f;
    float minTemp = 50.f;

    float minRadius = 0.f;
    float maxRadius = *std::max_element(m_distance_voxels.begin(), m_distance_voxels.end());

    // at the max radius, temp of max temp and interpoate to min temp
    for (int i = 0; i < m_distance_voxels.size(); i++) {
        float temp = 0.f;
        if (m_distance_voxels[i] == -1) {
            temp = 114.f;
        } else {
            temp = minTemp + (m_distance_voxels[i] / maxRadius) * (maxTemp - minTemp);
        }
        // std::cout << temp << std::endl;
        m_mock_gradient[i] = temp;
    }
}

std::vector<Vector3f>& Bread::calcGradient(std::vector<float> inputVec) {
    std::vector<Vector3f> gradVector;
    gradVector.resize(m_voxels.size());
    for (int x = 0; x < dimX; x++) {
        for (int y = 0; x < dimY; y++) {
            for (int z = 0; z < dimZ; z++) {
                int indexx1, indexx2;
                indicesToVoxel(std::max(x - 1, 0), y, z, indexx1);
                indicesToVoxel(std::min(x + 1, dimX - 1), y, z, indexx2);

                int indexy1, indexy2;
                indicesToVoxel(x, std::max(y - 1, 0), z, indexy1);
                indicesToVoxel(x, std::min(y + 1, dimY - 1), z, indexy2);

                int indexz1, indexz2;
                indicesToVoxel(x, y, std::max(z - 1, 0), indexz1);
                indicesToVoxel(x, y, std::min(z + 1, dimZ - 1), indexz2);

                float gradX = (inputVec[indexx2] - inputVec[indexx1]) / 2.f;
                float gradY = (inputVec[indexy2] - inputVec[indexy1]) / 2.f;
                float gradZ = (inputVec[indexz2] - inputVec[indexz1]) / 2.f;

                int index;
                indicesToVoxel(x, y, z, index);
                gradVector[index] = Vector3f(gradX, gradY, gradZ);
            }
        }
    }

    return gradVector;
}

std::vector<float>& Bread::gaussian(std::vector<float> inputVec) {

}

// for backmapping we somehow need to reverse solve for [u,v,w] in
// [x,y,z] = [u,v,w' + pg'[u,v,w] which idk how to do rn
void Bread::backmap(std::vector<Vector3f> grad) {
    std::vector<bool> deformedVoxels;
    // for each voxel
    // backmap to the original
    // perform trilinear or some interpolation to sample original voxles

    for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {
            for (int z = 0; z < dimZ; z++) {
                int index;
                indicesToVoxel(x, y, z, index);

                // get original location
                Vector3f newLoc = Vector3f(x, y, z);
                // approximate by subtracting the gradient here
                Vector3f oldLoc = newLoc - p * grad[index];

                // sample around old loc, just use nearest neighbor for now
                bool vox = voxelAt(int(oldLoc[0]), int(oldLoc[1]), int(oldLoc[2]));

                deformedVoxels[index] = vox;
            }
        }
    }
}

void Bread::forwardmap(std::vector<Vector3f> grad) {
    std::vector<bool> deformedVoxels;
    std::vector<bool> visited;
    // for each voxel
    // backmap to the original
    // perform trilinear or some interpolation to sample original voxles

    for (int u = 0; u < dimX; u++) {
        for (int v = 0; v < dimY; v++) {
            for (int w = 0; w < dimZ; w++) {
                int index;
                indicesToVoxel(u, v, w, index);

                // warp coordinate
                Vector3f newLoc = Vector3f(u, v, w) + p * grad[index];

                // value at old place
                bool vox = voxelAt(u, v, w);

                int newIndex;
                indicesToVoxel(int(newLoc[0]), int(newLoc[1]), int(newLoc[2]), newIndex);
                deformedVoxels[newIndex] = vox;
            }
        }
    }

    // maybe loop through visited and interpolate to fill in the gaps (hacky?)
}
