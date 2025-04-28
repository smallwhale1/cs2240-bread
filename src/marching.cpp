#include "marching.h"
#include <unordered_map>
#include <fstream>
#include <stdexcept>

using namespace std;
using namespace Eigen;

// Helper: safe voxel lookup
inline bool getVoxel(const vector<bool>& voxels, int x, int y, int z, int dimX, int dimY, int dimZ) {
    if (x < 0 || y < 0 || z < 0 || x >= dimX || y >= dimY || z >= dimZ) return false;
    return voxels[x + dimX * (y + dimY * z)];
}

// Helper: midpoint interpolation
Vector3f interpolate(const Vector3f& p1, const Vector3f& p2) {
    return 0.5f * (p1 + p2);
}

// Hashing for Vector3f
struct Vector3fHash {
    size_t operator()(const Vector3f& v) const {
        size_t h1 = hash<float>()(v.x());
        size_t h2 = hash<float>()(v.y());
        size_t h3 = hash<float>()(v.z());
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};

struct Vector3fEqual {
    bool operator()(const Vector3f& a, const Vector3f& b) const {
        return a.isApprox(b, 1e-5f); // Allow small floating point tolerance
    }
};

void marchingCubes(
    const vector<bool>& voxels,
    int dimX, int dimY, int dimZ,
    vector<Vector3f>& outVertices,
    vector<Triangle>& outTriangles,
    const int edgeTable[256],
    const int triTable[256][16]
    ) {
    const Vector3f vertexOffset[8] = {
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},
        {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}
    };

    const int edgeVertex[12][2] = {
        {0,1}, {1,2}, {2,3}, {3,0},
        {4,5}, {5,6}, {6,7}, {7,4},
        {0,4}, {1,5}, {2,6}, {3,7}
    };

    unordered_map<Vector3f, int, Vector3fHash, Vector3fEqual> vertexMap;

    for (int z = 0; z < dimZ - 1; ++z) {
        for (int y = 0; y < dimY - 1; ++y) {
            for (int x = 0; x < dimX - 1; ++x) {
                int cubeIndex = 0;
                bool cube[8];
                for (int i = 0; i < 8; ++i) {
                    cube[i] = getVoxel(voxels, x + vertexOffset[i].x(), y + vertexOffset[i].y(), z + vertexOffset[i].z(), dimX, dimY, dimZ);
                    if (cube[i]) cubeIndex |= (1 << i);
                }

                if (edgeTable[cubeIndex] == 0)
                    continue;

                Vector3f vertList[12];
                for (int i = 0; i < 12; ++i) {
                    if (edgeTable[cubeIndex] & (1 << i)) {
                        Vector3f p1 = Vector3f(x, y, z) + vertexOffset[edgeVertex[i][0]];
                        Vector3f p2 = Vector3f(x, y, z) + vertexOffset[edgeVertex[i][1]];
                        vertList[i] = interpolate(p1, p2);
                    }
                }

                for (int i = 0; triTable[cubeIndex][i] != -1; i += 3) {
                    int idx[3];
                    for (int j = 0; j < 3; ++j) {
                        const Vector3f& v = vertList[triTable[cubeIndex][i+j]];
                        auto it = vertexMap.find(v);
                        if (it != vertexMap.end()) {
                            idx[j] = it->second;
                        } else {
                            idx[j] = outVertices.size();
                            outVertices.push_back(v);
                            vertexMap[v] = idx[j];
                        }
                    }
                    outTriangles.push_back({ idx[0], idx[1], idx[2] });
                }
            }
        }
    }
}

void saveOBJ(const string& filename, const vector<Vector3f>& vertices, const vector<Triangle>& triangles) {
    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open OBJ file for writing!");
    }

    for (const auto& v : vertices)
        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";

    for (const auto& t : triangles)
        file << "f "
             << (t.v0 + 1) << " "
             << (t.v1 + 1) << " "
             << (t.v2 + 1) << "\n";
}
