#include "bread.h"
#include <iostream>
#include <algorithm>

using namespace Eigen;

void Bread::constructMockTemp() {
    // std::vector<float> mockGradients;
    m_mock_temp.resize(m_distance_voxels.size());

    float maxTemp = 110.f;
    float minTemp = 50.f;

    float minRadius = 0.f;
    float maxRadius = *std::max_element(m_distance_voxels.begin(), m_distance_voxels.end());

    // at the max radius, temp of max temp and interpoate to min temp
    for (int i = 0; i < m_distance_voxels.size(); i++) {
        float temp = 0.f;
        if (m_distance_voxels[i] == -1) {
            temp = 114.f;
        } else {
            // if (m_distance_voxels[i] <= 3) {
            //     temp = 110;
            // } else {

                temp = minTemp + ((maxRadius - m_distance_voxels[i]) / maxRadius) * (maxTemp - minTemp);
                std::cout << "distance: " << m_distance_voxels[i] << std::endl;
                std::cout << "temp:  " << temp << std::endl;
            // }
        }
        m_mock_temp[i] = temp;
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

// TODO: revisit
void Bread::convolveGaussian() {
    std::vector<float> tempCopy = m_mock_temp;

    for (int idx = 0; idx < m_mock_temp.size(); idx++) {
        m_mock_temp[idx] = 0;
        int x, y, z;
        voxelToIndices(idx, x, y, z);
        for (int i = -1 * m_filterRadius; i <= m_filterRadius; i++) {
            for (int j = -1 * m_filterRadius; j <= m_filterRadius; j++) {
                float temp = 0;
                for (int k = -1 * m_filterRadius; k <= m_filterRadius; k++) {
                    int idx_prime;
                    int x_prime = fmax(0, fmin(x + k, dimX - 1));
                    int y_prime = fmax(0, fmin(y + j, dimY - 1));
                    int z_prime = fmax(0, fmin(z + i, dimZ - 1));
                    indicesToVoxel(x_prime, y_prime, z_prime, idx_prime);

                    temp += tempCopy[idx_prime] * m_gaussianKernel[k];
                }
                m_mock_temp[idx] += temp / 3;
            }
        }

        for (int i = -1 * m_filterRadius; i <= m_filterRadius; i++) {
            for (int j = -1 * m_filterRadius; j <= m_filterRadius; j++) {
                float temp = 0;
                for (int k = -1 * m_filterRadius; k <= m_filterRadius; k++) {
                    int idx_prime;
                    int x_prime = fmax(0, fmin(x + i, dimX - 1));
                    int y_prime = fmax(0, fmin(y + k, dimY - 1));
                    int z_prime = fmax(0, fmin(z + j, dimZ - 1));
                    indicesToVoxel(x_prime, y_prime, z_prime, idx_prime);

                    temp += tempCopy[idx_prime] * m_gaussianKernel[k];
                }
                m_mock_temp[idx] += temp / 3;
            }
        }

        for (int i = -1 * m_filterRadius; i <= m_filterRadius; i++) {
            for (int j = -1 * m_filterRadius; j <= m_filterRadius; j++) {
                float temp = 0;
                for (int k = -1 * m_filterRadius; k <= m_filterRadius; k++) {
                    int idx_prime;
                    int x_prime = fmax(0, fmin(x + j, dimX - 1));
                    int y_prime = fmax(0, fmin(y + i, dimY - 1));
                    int z_prime = fmax(0, fmin(z + k, dimZ - 1));
                    indicesToVoxel(x_prime, y_prime, z_prime, idx_prime);

                    temp += tempCopy[idx_prime] * m_gaussianKernel[k];
                }
                m_mock_temp[idx] += temp / 3;
            }
        }


        std::cout << tempCopy[idx] << std::endl;
        m_mock_temp[idx] /= 9.f;
        std::cout << m_mock_temp[idx] << std::endl;
    }
}

void Bread::warpBubbles(std::vector<Vector3f> grad) {
    std::vector<bool> deformedVoxels;
    deformedVoxels.assign(m_voxels.size(), 0);
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
                Vector3f newLoc = Vector3f(u, v, w) + (p * grad[index]);

                int newIndex;
                indicesToVoxel(int(newLoc[0]), int(newLoc[1]), int(newLoc[2]), newIndex);

                if (newLoc[0] < 0 || newLoc[0] >= dimX || newLoc[1] < 0 || newLoc[1] >= dimY || newLoc[2] < 0 || newLoc[2] >= dimZ) {
                    deformedVoxels[index] = 0;
                } else {
                    deformedVoxels[index] = m_voxels[newIndex];
                }
            }
        }
    }

    for (int i = 0; i < m_voxels.size(); i++) {
        m_voxels[i] = deformedVoxels[i];
    }
}
