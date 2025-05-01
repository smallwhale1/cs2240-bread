#include "bread.h"
#include <iostream>
#include <algorithm>
#include <fstream>

using namespace std;

bool isInside(int x, int y, int z, int dimX, int dimY, int dimZ) {
    return x >= 0 && y >= 0 && z >= 0 && x < dimX && y < dimY && z < dimZ;
}

void Bread::extractVoxelSurfaceToOBJ(const vector<bool>& m_voxels, int dimX, int dimY, int dimZ, const string& filename) {
    ofstream obj(filename);
    if (!obj.is_open()) {
        cerr << "Failed to open output file\n";
        return;
    }

    vector<array<float, 3>> vertices;
    vector<array<int, 3>> faces;

    auto emit_face = [&](int x, int y, int z, int face) {
        // Face layout (quads made of 2 triangles)
        // 0: -X, 1: +X, 2: -Y, 3: +Y, 4: -Z, 5: +Z

        // Cube vertex offsets
        const float vx[8][3] = {
            {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0},  // bottom (z=0)
            {0,0,1}, {1,0,1}, {1,1,1}, {0,1,1}   // top (z=1)
        };

        // Faces as quads of 4 vertex indices into vx
        const int faceVerts[6][4] = {
            {0, 3, 7, 4},  // -X
            {1, 5, 6, 2},  // +X
            {0, 1, 5, 4},  // -Y
            {3, 2, 6, 7},  // +Y
            {0, 1, 2, 3},  // -Z
            {4, 5, 6, 7},  // +Z
        };

        int vStart = vertices.size();

        for (int i = 0; i < 4; ++i) {
            int vi = faceVerts[face][i];
            float vx_i = x + vx[vi][0];
            float vy_i = y + vx[vi][1];
            float vz_i = z + vx[vi][2];
            vertices.push_back({vx_i, vy_i, vz_i});
        }

        // Add 2 triangles (OBJ is 1-based)
        faces.push_back({vStart + 1, vStart + 2, vStart + 3});
        faces.push_back({vStart + 1, vStart + 3, vStart + 4});
    };

    for (int z = 0; z < dimZ; ++z) {
        for (int y = 0; y < dimY; ++y) {
            for (int x = 0; x < dimX; ++x) {
                if (!m_voxels[toIndex(x, y, z, dimX, dimY)]) continue;

                // Check each of the 6 faces
                const int dx[6] = {-1, 1,  0, 0,  0, 0};
                const int dy[6] = { 0, 0, -1, 1,  0, 0};
                const int dz[6] = { 0, 0,  0, 0, -1, 1};

                for (int f = 0; f < 6; ++f) {
                    int nx = x + dx[f];
                    int ny = y + dy[f];
                    int nz = z + dz[f];

                    bool isOutside = !isInside(nx, ny, nz, dimX, dimY, dimZ) ||
                                     !m_voxels[toIndex(nx, ny, nz, dimX, dimY)];

                    if (isOutside) {
                        emit_face(x, y, z, f);
                    }
                }
            }
        }
    }

    // Write to OBJ
    for (auto& v : vertices)
        obj << "v " << v[0] << " " << v[1] << " " << v[2] << "\n";
    for (auto& f : faces)
        obj << "f " << f[0] << " " << f[1] << " " << f[2] << "\n";

    obj.close();
}
