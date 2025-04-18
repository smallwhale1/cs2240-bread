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
    void indicesToVoxel(int x, int y, int z, int &index);
    void generateSphere(int x, int y, int z, int radius);
    void generateBubbles(int minRadius, int maxRadius);
    void fillIn();
    void writeBinvox(const std::string& filename, int dimX, int dimY, int dimZ, const std::vector<bool>& voxels, float translateX, float translateY, float translateZ, float scale);

    std::vector<float> m_temperatures;
    std::vector<float> m_W;
    std::vector<float> m_p;
    float timestep = 1.f; // maybe should be like 30??
    int bakingIterations = 20;
    void initTemperatures();
    void bake();
    void initBake();
    float prevDensity;

};

#endif // BREAD_H
