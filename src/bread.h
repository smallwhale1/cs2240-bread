#ifndef BREAD_H
#define BREAD_H
#include "Eigen/Dense"

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

    // parameters
    // temperature deformation
    float p = 1;
    // rising
    float S = 5;

    // deformation
    void forwardmap(std::vector<Eigen::Vector3f> grad);
    std::vector<bool> backmap(std::vector<Eigen::Vector3f> grad);
    void constructMockTemp();
    // std::vector<Eigen::Vector3f> calcGradient(std::vector<float> inputVec);
    std::vector<Eigen::Vector3f> calcGradient(int index, int kernel_size);
    // std::vector<float> gaussian(std::vector<float> inputVec);
    void generateGaussianFilter();
    std::vector<float> m_mock_temp;

    int m_filterRadius = 3; // change radius of filter
    std::vector<float> m_gaussianKernel;

    // rising
    // stores max bubble radius at a particular voxel
    std::vector<int> m_P;
};

#endif // BREAD_H
