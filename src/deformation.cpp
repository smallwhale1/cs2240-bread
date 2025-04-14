#include "bread.h"
#include <iostream>

using namespace Eigen;

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
                // TODO: fix this to actually inverse
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
