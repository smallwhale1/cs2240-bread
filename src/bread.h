#ifndef BREAD_H
#define BREAD_H

#include <vector>

class Bread {
private:

public:
    Bread();
    void init();

private:
    std::vector<bool> m_voxels;
    int dimX, dimY, dimZ;
    float translateX = 0.f;
    float translateY = 0.f;
    float translateZ = 0.f;
    float scale = 1.f;
    std::vector<float> m_distance_voxels;
    bool voxelAt(int x, int y, int z);
    void distanceVoxels();
    void voxelToSpatialCoords(int x, int y, int z, float &worldX, float &worldY, float &worldZ);
    void voxelToIndices(int index, int &x, int &y, int &z);
};

#endif // BREAD_H
