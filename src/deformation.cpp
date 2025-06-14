#include "bread.h"
#include <iostream>
#include <algorithm>

using namespace Eigen;
using namespace std;

void Bread::constructMockTemp() {
    m_mock_temp.resize(m_distance_voxels.size());

    float crustTemp = 110.f;
    float outerTemp = 110.f;
    float centerTemp = 50.f;

    float minRadius = 0.f;
    float maxRadius = *std::max_element(m_distance_voxels.begin(), m_distance_voxels.end());

    float effectiveMaxRadius = maxRadius - m_crust_thickness;

    for (int i = 0; i < m_distance_voxels.size(); i++) {
        float temp = 0.f;
        if (m_distance_voxels[i] == -1) {
            temp = outerTemp;
        } else if (m_distance_voxels[i] <= m_crust_thickness) {
            temp = crustTemp;
        } else {
            float distPastCrust = m_distance_voxels[i] - m_crust_thickness;
            temp = centerTemp + ((effectiveMaxRadius - distPastCrust) / effectiveMaxRadius) * (crustTemp - centerTemp);

            temp = std::min(std::max(temp, centerTemp), crustTemp);
        }
        m_mock_temp[i] = temp;
    }
}

void Bread::constructTemp() {
    m_temp.resize(m_distance_voxels.size());

    float outerTemp = 114.f; // not sure exactly what to make this

    for (int i = 0; i < m_distance_voxels.size(); i++) {
        float temp = 0.f;
        if (m_distance_voxels[i] == 0.f) {
            temp = outerTemp;
        } else {
            temp = m_temperatures[m_distance_voxels[i]];
        }
        m_temp[i] = temp;
    }
}

std::vector<Vector3f> Bread::calcGradient(std::vector<float> inputVec) {
    std::vector<Vector3f> gradVector;
    gradVector.resize(m_voxels.size());
    for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {
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
                gradVector[index] = Vector3f(gradX, gradY, gradZ).normalized();
            }
        }
    }

    return gradVector;
}

// std::vector<std::vector<float>> Bread::calcGradient(int index) {
//     int x, y, z;
//     voxelToIndices(index, x, y, z);

//     // get gradients in all three dimensions
//     std::vector<float> gradientKernelX;
//     std::vector<float> gradientKernelY;
//     std::vector<float> gradientKernelZ;
//     gradientKernelX.assign(pow(m_filterRadius * 2 + 1, 3), 0);
//     gradientKernelY.assign(pow(m_filterRadius * 2 + 1, 3), 0);
//     gradientKernelZ.assign(pow(m_filterRadius * 2 + 1, 3), 0);

//     int kernelIndex = 0;
//     for (int i = -1 * m_filterRadius; i <= m_filterRadius; i++) {
//         for (int j = -1 * m_filterRadius; j <= m_filterRadius; j++) {
//             for (int k = -1 * m_filterRadius; k <= m_filterRadius; k++) {
//                 int idx_prime;
//                 int x_prime = x + i;
//                 int y_prime = y + j;
//                 int z_prime = z + k;
//                 indicesToVoxel(x_prime, y_prime, z_prime, idx_prime);

//                 // create X gradient kernel
//                 int idx_next;
//                 int x_next = x_prime + 1;
//                 int y_next = y_prime;
//                 int z_next = z_prime;
//                 indicesToVoxel(x_next, y_next, z_next, idx_next);

//                 // bounds checking. TODO check: >= or > dimD depends on how dims are parsed. TODO: verify
//                 if (x_prime < 0 || x_prime >= dimX || y_prime < 0 || y_prime >= dimY || z_prime < 0 || z_prime >= dimZ) {
//                     gradientKernelX[kernelIndex] = m_mock_temp[fmax(0, fmin(idx_prime, dimX - 1))];
//                 } else if (x_next < 0 || x_next >= dimX || y_next < 0 || y_next >= dimY || z_next < 0 || z_next >= dimZ) {
//                     gradientKernelX[kernelIndex] = -m_mock_temp[fmax(0, fmin(idx_next, dimX - 1))];
//                 } else {
//                     gradientKernelX[kernelIndex] = m_mock_temp[idx_next] - m_mock_temp[idx_prime];
//                 }

//                 // create Y gradient kernel
//                 x_next = x_prime;
//                 y_next = y_prime + 1;
//                 z_next = z_prime;
//                 indicesToVoxel(x_next, y_next, z_next, idx_next);

//                 // bounds checking. TODO check: >= or > dimD depends on how dims are parsed. TODO: verify
//                 if (x_prime < 0 || x_prime >= dimX || y_prime < 0 || y_prime >= dimY || z_prime < 0 || z_prime >= dimZ) {
//                     gradientKernelY[kernelIndex] = m_mock_temp[fmax(0, fmin(idx_prime, dimY - 1))];
//                 } else if (x_next < 0 || x_next >= dimX || y_next < 0 || y_next >= dimY || z_next < 0 || z_next >= dimZ) {
//                     gradientKernelY[kernelIndex] = -m_mock_temp[fmax(0, fmin(idx_next, dimY - 1))];
//                 } else {
//                     gradientKernelY[kernelIndex] = m_mock_temp[idx_next] - m_mock_temp[idx_prime];
//                 }

//                 // create Z gradient kernel
//                 x_next = x_prime;
//                 y_next = y_prime;
//                 z_next = z_prime + 1;
//                 indicesToVoxel(x_next, y_next, z_next, idx_next);

//                 // bounds checking. TODO check: >= or > dimD depends on how dims are parsed. TODO: verify
//                 if (x_prime < 0 || x_prime >= dimX || y_prime < 0 || y_prime >= dimY || z_prime < 0 || z_prime >= dimZ) {
//                     gradientKernelZ[kernelIndex] = m_mock_temp[fmax(0, fmin(idx_prime, dimZ - 1))];
//                 } else if (x_next < 0 || x_next >= dimX || y_next < 0 || y_next >= dimY || z_next < 0 || z_next >= dimZ) {
//                     gradientKernelZ[kernelIndex] = -m_mock_temp[fmax(0, fmin(idx_next, dimZ - 1))];
//                 } else {
//                     gradientKernelZ[kernelIndex] = m_mock_temp[idx_next] - m_mock_temp[idx_prime];
//                 }

//                 // // sanity bounds check. TODO: verify
//                 // if (idx < 0 || idx > m_voxels.size()) {
//                 //     continue;
//                 // }

//                 kernelIndex++;
//             }
//         }
//     }

//     std::vector<std::vector<float>> gradients;
//     gradients.push_back(gradientKernelX);
//     gradients.push_back(gradientKernelY);
//     gradients.push_back(gradientKernelZ);
//     return gradients;
// }

// TODO DELETE THIS COMMENT: used my code from filter so uhh i hope it's right LOL
// TODO: call this function somewhere in init so we create our filter before we sim any geometry changes
// creates a 1D filter, so we convolve with this over all three dims
void Bread::generateGaussianFilter() {
    m_gaussianKernel.assign(m_filterRadius * 2 + 1, 0);

    float stddev = pow(m_filterRadius / 3.f, 2);

    int kernel_index = -1;
    float total_weight = 0;
    for (int i = -1 * m_filterRadius; i <= m_filterRadius; i++) {
        kernel_index++;
        m_gaussianKernel.at(i + m_filterRadius) = pow(exp(1), (-1 * pow(i, 2)) / (2 * stddev)) / sqrt(2 * M_PI * stddev);
        total_weight += m_gaussianKernel.at(kernel_index);
    }

    kernel_index = -1;
    for (int i = -1 * m_filterRadius; i <= m_filterRadius; i++) {
        kernel_index++;
        m_gaussianKernel.at(kernel_index) /= total_weight;
    }
}

void Bread::convolveGaussian() {
    std::vector<Eigen::Vector3f> tempCopy = m_gradVector;

    // x
    for (int idx = 0; idx < m_gradVector.size(); idx++) {
        int x, y, z;
        voxelToIndices(idx, x, y, z);
        Eigen::Vector3f smoothed = Eigen::Vector3f::Zero();

        for (int k = -m_filterRadius; k <= m_filterRadius; k++) {
            int xk = std::clamp(x + k, 0, dimX - 1);
            int idxPrime;
            indicesToVoxel(xk, y, z, idxPrime);
            smoothed += m_gaussianKernel[k + m_filterRadius] * tempCopy[idxPrime];
        }

        m_gradVector[idx] = smoothed;
    }

    tempCopy = m_gradVector;

    // y
    for (int idx = 0; idx < m_gradVector.size(); idx++) {
        int x, y, z;
        voxelToIndices(idx, x, y, z);
        Eigen::Vector3f smoothed = Eigen::Vector3f::Zero();

        for (int k = -m_filterRadius; k <= m_filterRadius; k++) {
            int yk = std::clamp(y + k, 0, dimY - 1);
            int idxPrime;
            indicesToVoxel(x, yk, z, idxPrime);
            smoothed += m_gaussianKernel[k + m_filterRadius] * tempCopy[idxPrime];
        }

        m_gradVector[idx] = smoothed;
    }

    tempCopy = m_gradVector;

    // z
    for (int idx = 0; idx < m_gradVector.size(); idx++) {
        int x, y, z;
        voxelToIndices(idx, x, y, z);
        Eigen::Vector3f smoothed = Eigen::Vector3f::Zero();

        for (int k = -m_filterRadius; k <= m_filterRadius; k++) {
            int zk = std::clamp(z + k, 0, dimZ - 1);
            int idxPrime;
            indicesToVoxel(x, y, zk, idxPrime);
            smoothed += m_gaussianKernel[k + m_filterRadius] * tempCopy[idxPrime];
        }

        m_gradVector[idx] = smoothed;
    }
}

float Bread::trilinearSampleVoxel(float x, float y, float z, std::vector<bool>& inputVec) {
    int x0 = floor(x);
    int x1 = x0 + 1;
    int y0 = floor(y);
    int y1 = y0 + 1;
    int z0 = floor(z);
    int z1 = z0 + 1;

    if (x0 < 0 || x1 >= dimX || y0 < 0 || y1 >= dimY || z0 < 0 || z1 >= dimZ)
        return 0.f;

    auto getVoxFloat = [&](int xi, int yi, int zi) -> float {
        int idx;
        indicesToVoxel(xi, yi, zi, idx);
        return inputVec[idx] ? 1.0f : 0.0f;
    };

    float xd = x - x0;
    float yd = y - y0;
    float zd = z - z0;

    // 8 corners of voxel
    float c000 = getVoxFloat(x0, y0, z0);
    float c001 = getVoxFloat(x0, y0, z1);
    float c010 = getVoxFloat(x0, y1, z0);
    float c011 = getVoxFloat(x0, y1, z1);
    float c100 = getVoxFloat(x1, y0, z0);
    float c101 = getVoxFloat(x1, y0, z1);
    float c110 = getVoxFloat(x1, y1, z0);
    float c111 = getVoxFloat(x1, y1, z1);

    // interp z
    float c00 = c000 * (1 - zd) + c001 * zd;
    float c01 = c010 * (1 - zd) + c011 * zd;
    float c10 = c100 * (1 - zd) + c101 * zd;
    float c11 = c110 * (1 - zd) + c111 * zd;

    // interp y
    float c0 = c00 * (1 - yd) + c01 * yd;
    float c1 = c10 * (1 - yd) + c11 * yd;

    // interp x
    return c0 * (1 - xd) + c1 * xd;
}

std::vector<bool> Bread::warpBubbles(std::vector<Vector3f> grad) {
    std::vector<bool> deformedVoxels;
    deformedVoxels.assign(m_voxels.size(), 0);
    std::vector<bool> visited;
    // for each voxel
    // backmap to the original
    // perform trilinear or some interpolation to sample original voxels

    for (int u = 0; u < dimX; u++) {
        for (int v = 0; v < dimY; v++) {
            for (int w = 0; w < dimZ; w++) {
                int index;
                indicesToVoxel(u, v, w, index);

                // warp coordinate
                Vector3f newLoc = Vector3f(u, v, w) - (p * grad[index]);

                int newIndex;
                indicesToVoxel(int(newLoc[0]), int(newLoc[1]), int(newLoc[2]), newIndex);

                if (newLoc[0] < 0 || newLoc[0] >= dimX || newLoc[1] < 0 || newLoc[1] >= dimY || newLoc[2] < 0 || newLoc[2] >= dimZ) {
                    deformedVoxels[index] = 0;
                } else {
                    // deformedVoxels[index] = m_voxels[newIndex];

                    float sample = trilinearSampleVoxel(newLoc[0], newLoc[1], newLoc[2], m_voxels);
                    deformedVoxels[index] = sample > 0.5f;
                }
            }
        }
    }

    return deformedVoxels;

    // for (int i = 0; i < m_voxels.size(); i++) {
    //     m_voxels[i] = deformedVoxels[i];
    // }
}
std::vector<bool> Bread::rise(std::vector<Vector3f> grad, std::vector<bool> inputVec, float scaleAmt, float scaleAmtY) {
    std::vector<bool> deformedVoxels(m_voxels.size(), false);
    cout << "scale: " << 1.0 + scaleAmt << endl;

    for (int u = 0; u < dimX; u++) {
        for (int v = 0; v < dimY; v++) {
            for (int w = 0; w < dimZ; w++) {

                int originalInd;
                indicesToVoxel(u, v, w, originalInd);

                float scaleFactor = (1.0 + scaleAmt);
                float scaleFactorY = (1.0 + scaleAmtY);

                float worldX;
                float worldY;
                float worldZ;

                voxelToSpatialCoords(u, v, w, worldX, worldY, worldZ);

                worldX /= scaleFactor;
                worldY /= scaleFactorY;
                // worldY /= (scaleFactor * 1.05);
                worldZ /= scaleFactor;

                int newX;
                int newY;
                int newZ;

                spatialToVoxel(worldX, worldY, worldZ, newX, newY, newZ);

                Vector3f xyz = Vector3f(newX, newY, newZ);

                int xyzX = static_cast<int>(xyz[0]);
                int xyzY = static_cast<int>(xyz[1]);
                int xyzZ = static_cast<int>(xyz[2]);

                if (xyzX < 0 || xyzX >= dimX ||
                    xyzY < 0 || xyzY >= dimY ||
                    xyzZ < 0 || xyzZ >= dimZ) {
                    continue;
                }

                int newIndex;
                indicesToVoxel(xyzX, xyzY, xyzZ, newIndex);

                if (newIndex < 0 || newIndex >= m_voxels.size()) {
                    continue;
                }

                // deformedVoxels[originalInd] = inputVec[newIndex];
                // float sample = trilinearSampleVoxel(xyzX, xyzY, xyzZ, inputVec);
                // deformedVoxels[originalInd] = sample > 0.5f;
                // deformedVoxels[originalInd] = m_voxels[newIndex];
                deformedVoxels[originalInd] = m_voxels[newIndex];
                // if (deformedVoxels[originalInd] == 1) {
                //     for (int yy = dimY; yy > xyzY; yy--) {
                //         int yyInd;
                //         int yyIndLess;
                //         indicesToVoxel(xyzX, yy, xyzZ, yyInd);
                //         indicesToVoxel(xyzX, yy - 1, xyzZ, yyIndLess);
                //         deformedVoxels[yyInd] = deformedVoxels[yyIndLess];
                //     }
                // }

            }
        }
    }

    return deformedVoxels;
}

// void Bread::rise(std::vector<Vector3f> grad) {
//     std::vector<bool> deformedVoxels(m_voxels.size(), false);

//     for (int x = 0; x < dimX; x++) {
//         for (int y = 0; y < dimY; y++) {
//             for (int z = 0; z < dimZ; z++) {
//                 int index;
//                 indicesToVoxel(x, y, z, index);

//                 if (index < 0 || index >= m_voxels.size())
//                     continue;

//                 Vector3f rst = Vector3f(x, y, z) / (S * m_P[index]);

//                 int rstX = static_cast<int>(rst[0]);
//                 int rstY = static_cast<int>(rst[1]);
//                 int rstZ = static_cast<int>(rst[2]);

//                 if (rstX < 0 || rstX >= dimX ||
//                     rstY < 0 || rstY >= dimY ||
//                     rstZ < 0 || rstZ >= dimZ)
//                     continue;

//                 int rstIndex;
//                 indicesToVoxel(rstX, rstY, rstZ, rstIndex);

//                 if (rstIndex < 0 || rstIndex >= grad.size())
//                     continue;

//                 Vector3f uvw = rst - m_P[rstIndex] * grad[rstIndex];

//                 int uvwX = static_cast<int>(uvw[0]);
//                 int uvwY = static_cast<int>(uvw[1]);
//                 int uvwZ = static_cast<int>(uvw[2]);

//                 if (uvwX < 0 || uvwX >= dimX ||
//                     uvwY < 0 || uvwY >= dimY ||
//                     uvwZ < 0 || uvwZ >= dimZ)
//                     continue;

//                 int newIndex;
//                 indicesToVoxel(uvwX, uvwY, uvwZ, newIndex);

//                 if (newIndex < 0 || newIndex >= m_voxels.size())
//                     continue;

//                 deformedVoxels[index] = m_voxels[newIndex];
//             }
//         }
//     }

//     m_voxels = std::move(deformedVoxels);
// }
