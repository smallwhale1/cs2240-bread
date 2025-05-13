#include "bread.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "marching.h"
#include <algorithm>

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QFileInfo>
#include <cmath>
#include <opencv2/opencv.hpp>

using namespace std;

Bread::Bread() {
}

int Bread::toIndex(int x, int y, int z, int dimX, int dimY) {
    return x + y * dimX + z * dimX * dimY;
}

void Bread::init() {

    // Specify voxel filepath
    const std::string& filepath = "meshes-binvox/bun-128.binvox";
    // const std::string& filepath = "frames-48/128-rise-47.binvox";
    // const std::string& filepath = "256-original.binvox";

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file");
    }

    // Load in voxel into bool vector
    std::string line;
    std::getline(file, line);
    if (line != "#binvox 1") {
        throw std::runtime_error("Invalid header");
    }

    while (std::getline(file, line)) {
        if (line.substr(0, 3) == "dim") {
            sscanf(line.c_str(), "dim %d %d %d", &dimX, &dimY, &dimZ);
        } else if (line.substr(0, 10) == "translate ") {
            // Handle translate:
            sscanf(line.c_str(), "translate %f %f %f", &translateX, &translateY, &translateZ);
        } else if (line.substr(0, 6) == "scale ") {
            // Handle scale:
            sscanf(line.c_str(), "scale %f", &scale);
        } else if (line == "data") {
            // End of header:
            break;
        }
    }

    int numVoxels = dimX * dimY * dimZ;
    m_voxels.resize(numVoxels);

    // Parse data
    unsigned char value = 0;
    unsigned char count = 0;
    int index = 0;

    while (file.read(reinterpret_cast<char*>(&value), 1) &&
           file.read(reinterpret_cast<char*>(&count), 1)) {

        for (int i = 0; i < count; ++i) {
            m_voxels[index++] = (value != 0);
        }
        auto x = 1;
    }
    file.close();

    // Add padding around the edges to allow for rising
    addPadding(10);
    m_P.resize(m_voxels.size());
    std::fill(m_P.begin(), m_P.end(), 1.0);
    fillIn();
    distanceVoxels();

    // Generate bubbles
    generateSphere(0, 0, 0, 2);
    generateBubbles(1, 8); // baguette
    // generateBubbles(1, 5); // sourdough
    // generateBubbles(0, 2); // sandwich

    const std::string pFile = "P-128.bin";
    saveP(pFile);

    // For cross sections
    // voxelCopy = m_voxels;
    // for (int i = 0; i < m_voxels.size(); i++) {
    //     int x, y, z;
    //     voxelToIndices(i, x, y, z);
    //     if (z < dimZ / 2) {
    //         voxelCopy[i] = 0;
    //     }
    // }
    // writeBinvox("128-original.binvox", dimX, dimY, dimZ, voxelCopy, translateX, translateY, translateZ, scale);

    initTemperatures();
    initBake();
    generateGaussianFilter();
    m_3d_temperatures.resize(m_voxels.size());

    for (int i = 0; i < bakingIterations; i++) {

        bake(i);
        fillTemps();
        m_gradVector = calcGradient(m_3d_temperatures);
        std::vector<bool> warped = warpBubbles(m_gradVector);
        std::vector<bool> risen = rise(m_gradVector, warped,
                                       (i + 1) * (S_change / bakingIterations),
                                       (i + 1) * (S_change_y / bakingIterations));

        m_frame++;
        for (int j = 0; j < m_voxels.size(); j++) {

            int x, y, z;
            voxelToIndices(j, x, y, z);

            if (z > dimZ / 48 * m_frame) {
                risen[j] = 0;
            }
        }

        std::string filename = "128-rise-" + std::to_string(m_frame) + ".binvox";
        writeBinvox(filename, dimX, dimY, dimZ, risen, translateX, translateY, translateZ, scale);
    }

    // To show heat map
    // heatMap();

    // Saving crust
    saveJPG();
    vector<Triangle> triangles;
    marchingCubes(m_voxels, dimX, dimY, dimZ, outVertices, triangles, edgeTable, triangleTable);
    saveOBJ("bread.obj", outVertices, triangles);
    saveMTL();
}

void Bread::saveP(const std::string& filepath) {

    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    size_t size = m_P.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(m_P.data()), size * sizeof(int));
    out.close();
}

void Bread::loadP(const std::string& filepath) {

    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    m_P.resize(size);
    in.read(reinterpret_cast<char*>(m_P.data()), size * sizeof(int));
    in.close();
}

void Bread::saveDistanceVoxels(const std::string& filepath) {

    std::ofstream out(filepath, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for writing: " + filepath);
    }

    size_t size = m_distance_voxels.size();
    out.write(reinterpret_cast<const char*>(&size), sizeof(size));
    out.write(reinterpret_cast<const char*>(m_distance_voxels.data()), size * sizeof(float));
    out.close();
}

void Bread::loadDistanceVoxels(const std::string& filepath) {

    std::ifstream in(filepath, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file for reading: " + filepath);
    }

    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size));
    m_distance_voxels.resize(size);
    in.read(reinterpret_cast<char*>(m_distance_voxels.data()), size * sizeof(float));
    in.close();
}

void Bread::distanceVoxels() {

    m_distance_voxels.resize(m_voxels.size(), std::numeric_limits<float>::max());
    std::queue<std::tuple<int, int, int>> q;

    for (int x = 0; x < dimX; ++x) {
        for (int y = 0; y < dimY; ++y) {
            for (int z = 0; z < dimZ; ++z) {
                int idx;
                indicesToVoxel(x, y, z, idx);
                if (!m_voxels[idx]) {
                    m_distance_voxels[idx] = 0.f;
                    continue;
                }

                bool isSurface = false;
                for (int dx = -1; dx <= 1 && !isSurface; dx++) {
                    for (int dy = -1; dy <= 1 && !isSurface; dy++) {
                        for (int dz = -1; dz <= 1 && !isSurface; dz++) {
                            if (dx == 0 && dy == 0 && dz == 0) {
                                continue;
                            }
                            int nextX = x + dx;
                            int nextY = y + dy;
                            int nextZ = z + dz;
                            if (nextX < 0 || nextY < 0 || nextZ < 0 || nextX >= dimX || nextY >= dimY || nextZ >= dimZ)
                                continue;
                            int nIdx;
                            indicesToVoxel(nextX, nextY, nextZ, nIdx);
                            if (!m_voxels[nIdx]) {
                                isSurface = true;
                                m_distance_voxels[idx] = 1.f;
                                q.emplace(x, y, z);
                            }
                        }
                    }
                }
            }
        }
    }

    const int dx[6] = {1, -1, 0, 0, 0, 0};
    const int dy[6] = {0, 0, 1, -1, 0, 0};
    const int dz[6] = {0, 0, 0, 0, 1, -1};

    while (!q.empty()) {
        auto [x, y, z] = q.front();
        q.pop();
        int currIdx;
        indicesToVoxel(x, y, z, currIdx);
        float currDist = m_distance_voxels[currIdx];

        for (int i = 0; i < 6; ++i) {
            int nextX = x + dx[i];
            int nextY = y + dy[i];
            int nextZ = z + dz[i];
            if (nextX < 0 || nextY < 0 || nextZ < 0 || nextX >= dimX || nextY >= dimY || nextZ >= dimZ) {
                continue;
            }
            int nIdx;
            indicesToVoxel(nextX, nextY, nextZ, nIdx);
            if (m_voxels[nIdx] && m_distance_voxels[nIdx] > currDist + 1) {
                m_distance_voxels[nIdx] = currDist + 1;
                q.emplace(nextX, nextY, nextZ);
            }
        }
    }
}


void Bread::addPadding(int paddingAmt) {
    int newDimX = dimX + paddingAmt * 2;
    int newDimY = dimY + paddingAmt * 2;
    int newDimZ = dimZ + paddingAmt * 2;

    vector<bool> newVoxels(newDimX * newDimY * newDimZ, false);

    for (int x = 0; x < newDimX; x++) {
        for (int y = 0; y < newDimY; y++) {
            for (int z = 0; z < newDimZ; z++) {
                int oldIndX = x - paddingAmt;
                int oldIndY = y - paddingAmt;
                int oldIndZ = z - paddingAmt;

                int newIndex = y + z * newDimZ + x * newDimX * newDimZ;

                if (oldIndX < 0 || oldIndY < 0 || oldIndZ < 0 ||
                    oldIndX >= dimX || oldIndY >= dimY || oldIndZ >= dimZ) {
                    newVoxels[newIndex] = 0;
                } else {
                    int oldIndex;
                    indicesToVoxel(oldIndX, oldIndY, oldIndZ, oldIndex);
                    newVoxels[newIndex] = m_voxels[oldIndex];
                }
            }
        }
    }

    m_voxels = std::move(newVoxels);

    dimX = newDimX;
    dimY = newDimY;
    dimZ = newDimZ;
}


void Bread::voxelToIndices(int index, int &x, int &y, int &z) {

    int a = dimX * dimY;
    z = index / a;

    int b = index - a * z;
    y = b / dimX;
    x = b % dimX;
}

void Bread::indicesToVoxel(int x, int y, int z, int &index) {
    index = y;
    index += z * dimZ;
    index += x * dimX * dimZ;
}

void Bread::voxelToSpatialCoords(int x, int y, int z, float &worldX, float &worldY, float &worldZ) {

    float x_n = (x + 0.5f) / dimX;
    float y_n = (y + 0.5f) / dimY;
    float z_n = (z + 0.5f) / dimZ;

    worldX = scale * x_n + translateX;
    worldY = scale * y_n + translateY;
    worldZ = scale * z_n + translateZ;
}

void Bread::spatialToVoxel(float worldX, float worldY, float worldZ, int &x, int &y, int &z) {

    float x_n = (worldX - translateX) / scale;
    float y_n = (worldY - translateY) / scale;
    float z_n = (worldZ - translateZ) / scale;

    x = static_cast<int>(std::floor(x_n * dimX));
    y = static_cast<int>(std::floor(y_n * dimY));
    z = static_cast<int>(std::floor(z_n * dimZ));

    x = std::clamp(x, 0, dimX - 1);
    y = std::clamp(y, 0, dimY - 1);
    z = std::clamp(z, 0, dimZ - 1);
}

bool Bread::voxelAt(int x, int y, int z) {
    return m_voxels[x * dimX * dimZ + z * dimZ + y];
}

void Bread::generateSphere(int x, int y, int z, int radius) {

    std::vector<float> boundingBox;
    int boundingBoxWidth = 2 * radius + 1;
    boundingBox.assign(pow(boundingBoxWidth, 3), 0);
    int count = 0;

    for (int i = -1 * radius; i <= radius; i++) {
        for (int j = -1 * radius; j <= radius; j++) {
            for (int k = -1 * radius; k <= radius; k++) {
                int idx;
                int x_prime = x + i;
                int y_prime = y + j;
                int z_prime = z + k;

                if (x_prime < 0 || x_prime >= dimX || y_prime < 0 || y_prime >= dimY || z_prime < 0 || z_prime >= dimZ) {
                    continue;
                }

                indicesToVoxel(x_prime, y_prime, z_prime, idx);

                if (idx < 0 || idx > m_voxels.size()) {
                    continue;
                }

                float distance = sqrt(std::pow(x - x_prime, 2) + std::pow(y - y_prime, 2) + std::pow(z - z_prime, 2));
                if (distance <= radius && m_distance_voxels[idx] >= m_crust_thickness) {
                    m_voxels[idx] = 0;
                    m_P[idx] = std::max(m_P[idx], radius);
                    count++;
                }
            }
        }
    }
}

// Use min of 2 and max of 20 here for baguette setting
void Bread::generateBubbles(int minRadius, int maxRadius) {
    int radius = minRadius;

    while (radius <= maxRadius) {
        // Subtract spheres of minRadius
        m_frame++;
        std::vector<bool> voxelCopy = m_voxels;
        for (int i = 0; i < m_voxels.size(); i++) {
            int x, y, z;
            voxelToIndices(i, x, y, z);
            if (z > dimZ / 48 * m_frame) {
                voxelCopy[i] = 0;
            }
        }

        writeBinvox("128-original-" + std::to_string(m_frame) + ".binvox", dimX, dimY, dimZ, voxelCopy, translateX, translateY, translateZ, scale);

        int numSpheres = k / pow(radius, d);
        for (int i = 0; i < numSpheres; i++) {
            // Generate random coord inside dims
            int x = arc4random_uniform(dimX);
            int y = arc4random_uniform(dimY);
            int z = arc4random_uniform(dimZ);
            generateSphere(x, y, z, radius);
        }
        // Increment
        radius++;
    }
}

void Bread::fillIn() {
    for (int x = 0; x < dimX; x++) {
        for (int y = 0; y < dimY; y++) {

            int state = 0;
            int startZ = -1;
            for (int z = 0; z < dimZ; z++) {
                if (state == 0) {
                    if (voxelAt(x, y, z)) {
                        state = 1;
                        startZ = z;
                    }
                } else {
                    if (voxelAt(x, y, z)) {
                        state = 0;
                        if (startZ + 1 < z) {
                            for (int w = startZ + 1; w < z; w++) {
                                int index;
                                indicesToVoxel(x, y, w, index);
                                m_voxels[index] = 1;
                            }
                        }
                    }
                }
            }
        }
    }
}

// Iterates over every distance for each time step, stores results in temperatures vector
void Bread::bake(int time){

    double hc = 0.5; // heat transfer coefficient for convection
    double sigma = 5.670374419e-8;
    double temp_air = 483.150; // temp in air
    double temp_surface = m_temperatures[0]; // starting temp at the surface of the dough
    double temp_radial = 2.0 * temp_air; // temp at heat source
    double emissivity_product = 0.9;
    double emissivity_radial = 0.5;
    double asp = 0.1; // length of the sample
    double bsp = 0.2; // width of the sample
    double lsp = 0.5; // distance between radial source and sample
    double a = asp / lsp;
    double b = bsp / lsp;
    double a1 = 1.0 + pow(a, 2);
    double b1 = 1.0 + pow(b, 2);

    // shape factor - compares shape of bread to shape of radial source
    double fsp = (2.0 / (M_PI * a * b)) * (
                    std::log( sqrt((a1 * b1) / (1.0 + pow(a, 2) + pow(b, 2)))) +
                    (a * sqrt(b1) * std::atan(a / sqrt(b1))) +
                    (b * sqrt(a1) * std::atan(b / sqrt(a1))) -
                    (a * std::atan(a)) - (b * std::atan(b))
                    );

    // heat transfer coefficient for radiation
    double hr = (sigma * (pow(temp_radial, 2) + pow(temp_surface, 2)) * (temp_radial + temp_surface)) /
               ((1.0 / emissivity_product) + (1.0 / emissivity_radial) - 2.0 + (1.0 / fsp));

    double k = 0.5; // thermal conductivity
    double lambda = 2.257; // latent heat of evaporation/vaporization of water
    double diffusivity = 1.35e-10; // liquid water diffusivity
    double specific_heat = 3500.0; // amount of heat required to increase the temperature of a specific material by one degree
    double distance = 0.005;
    double w_air = 0.0; // water content of the air

    std::vector<double> dtdt(m_temperatures.size(), 0.0); // change in temperature over time
    std::vector<double> dtdx(m_temperatures.size(), 0.0); // change in temperature over distance
    std::vector<double> dtdx2(m_temperatures.size(), 0.0); // change in roc of temprature over distance

    std::vector<double> dWdt(m_temperatures.size(), 0.0); // change in water diffusion over t
    std::vector<double> dWdx(m_temperatures.size(), 0.0); // change in water diffusion over x
    std::vector<double> dWdx2(m_temperatures.size(), 0.0); // derivative of dWdx

    // Fill in dWdx
    double h_w = 0.00140 * m_temperatures[0] + 0.27 * m_W[0] - 0.0004 * m_temperatures[0] * m_W[0] - 0.77 * m_W[0] * m_W[0];
    dWdx[0] = h_w * (m_W[0] - w_air);
    dWdx[m_W.size() - 1] = 0.f;
    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            continue;
        }
        dWdx[x] = (m_W[x+1] - m_W[x-1]) / (distance * 2.0);
    }

    // Fill in dWdx2
    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            if (x == 0) {
                dWdx2[x] = (dWdx[1] - dWdx[0]) / distance;
            } else if (x == m_temperatures.size() - 1) {
                dWdx2[x] = (dWdx[x] - dWdx[x - 1]) / distance;
            }
        } else {
            dWdx2[x] = (dWdx[x+1] - dWdx[x-1]) / (distance * 2.0);
        }
    }

    // Fill in dWdt
    for (int x = 0; x < m_W.size(); x++) {
        dWdt[x] = diffusivity * dWdx2[x];
    }

    // Timestep forward
    for (int x = 0; x < m_W.size(); x++) {
        m_W[x] += timestep * dWdt[x];
    }
    m_W[0] = m_W[1] - distance * dWdx[0];
    m_W[m_W.size() - 1] = m_W[m_W.size() - 2];

    // Fill in dtdx
    for(int x = 0; x < m_temperatures.size(); x++){
        if (x == m_temperatures.size() - 1) {
            dtdx[x] = 0.0;
        } else if (x == 0) {
            dtdx[x] = ((hr * (temp_radial - temp_surface)) + (hc * (temp_air - temp_surface)) - (lambda * m_p[x] * diffusivity * (dWdx[0]))) / (-k);
        } else {
            dtdx[x] = (m_temperatures[x + 1] - m_temperatures[x - 1]) / (distance * 2.0);
        }
    }

    // Fill in dtdx2
    for(int x = 0; x < dtdx2.size(); x++){
        if(x == 0){
            dtdx2[x] = (dtdx[1] - dtdx[0]) / distance;
        } else if (x == m_temperatures.size() - 1) {
            dtdx2[x] = (dtdx[x] - dtdx[x - 1]) / distance;
        } else {
            dtdx2[x] = (dtdx[x + 1] - dtdx[x - 1]) / (distance * 2.0);
        }
    }

    // Fill in dtdt
    for (int x = 0; x < dtdt.size(); x++) {
        double new_p = 170.0 + (284.0 * m_W[x]);
        double dpdt = (new_p - m_p[x]) / distance;
        m_p[x] = new_p;
        dtdt[x] = (k * dtdx2[x]) / (new_p * specific_heat) + (lambda * dWdt[x]) / specific_heat + (lambda * m_W[x] * dpdt) / (new_p * specific_heat);
    }

    // Update m_temperatures
    for (int x = 0; x < m_temperatures.size(); x++) {
        m_temperatures[x] += timestep * dtdt[x];
    }

    // Create crust
    createCrust(time, dWdt);
}

void Bread::initBake(){

    // Fill global variables
    m_W.assign(m_temperatures.size(), 0.4);
    m_p.reserve(m_temperatures.size());
    m_p.assign(m_temperatures.size(), 285.0);
    m_L.assign(m_distance_voxels.size(), 70.f);

}

void Bread::initTemperatures(){

    float largest = *std::max_element(m_distance_voxels.begin(), m_distance_voxels.end());
    m_temperatures.resize(int(largest));
    m_temperatures.assign(int(largest), 298.0);

    std::vector<float> m_L(m_temperatures.size(), 90.f);
    crust_time = 0.f;
}

void Bread::fillTemps() {
    for (int i = 0; i < m_voxels.size(); i++) {
        if (m_distance_voxels[i] <= m_crust_thickness) {
            m_3d_temperatures[i] = static_cast<float>(m_temperatures[0]);
        } else {
            int dist = static_cast<int>(m_distance_voxels[i]);
            m_3d_temperatures[i] = static_cast<float>(m_temperatures[dist]);
        }
    }
}

void Bread::heatMap() {
   std::vector<std::vector<double>> data;
   int y = dimY/2;
   int rows = dimZ;
   int cols = dimX;

   for (int i = rows - 1; i > -1; i--) {
       std::vector<double> row;
       for (int j = 0; j < cols; j++) {
           float dist = m_distance_voxels[j * dimX * dimZ + i * dimZ + y];
           if (dist > 0.f) {
               int idx = dist - 1;
               row.push_back(m_temperatures[idx] - 273.15);
           } else {
               row.push_back(145.0);
           }

           // if (voxelAt(j, y, i)) {
           //     row.push_back(1.0);
           // } else {
           //     row.push_back(0.0);
           // }
       }
       data.push_back(row);
   }

   cv::Mat mat(rows, cols, CV_32F);

   for (int i = 0; i < rows; ++i)
       for (int j = 0; j < cols; ++j)
           mat.at<float>(i, j) = data[i][j];

   cv::Mat normalized;
   cv::normalize(mat, normalized, 0, 255, cv::NORM_MINMAX);
   normalized.convertTo(normalized, CV_8U);

   cv::Mat colorMap;
   cv::applyColorMap(normalized, colorMap, cv::COLORMAP_JET);
   cv::imshow("Heatmap", colorMap);
   cv::waitKey(0);
}

void Bread::createCrust(int time, std::vector<double> dWdt){

    // Determine thickness of crust for this time step - based on number of voxels and time
    double crust_thickness = dimX * time * (0.03125 / bakingIterations);
    bool addToCrust = true;
    rgb_colors.clear();

    for(int i = 0; i < m_distance_voxels.size(); i++){

        // temp > 120C/393.15K, water activity < 0.6
        if(m_distance_voxels[i] < crust_thickness && m_distance_voxels[i] != 0 && m_temperatures[std::floor((m_distance_voxels[i]))] > 390.f && dWdt[i] < 0.6f){

            if (addToCrust) {
                crust_time ++;
                addToCrust = false;
            }
            //how long have we been updating the crust color, hopefuly goes for about 20 iterations

            double t1 = std::pow(7.923310f, 6.f);
            float water_activity = m_W[m_distance_voxels[i]];
            double t2 = (std::pow(2.739710f, 6.f) / water_activity);
            double temp1 = t1 + t2;
            double temp2 = -1 * ((std::pow(8.701510, 3) + (49.4738 / water_activity)) / (m_temperatures[0]));
            double k = temp1 * std::pow(M_E, temp2);

            k -= 34000;
            k /= 10000;
            m_L[i] += -k * m_L[i] * timestep / 60.f;

            float a = -4.5f + (.75 * crust_time);
            float b = 22.6 + (2.9f * crust_time);

            std::vector<float> temp = labToRgb(m_L[i], a, b);
            rgb_colors.push_back({(float)m_distance_voxels[i], temp[0], temp[1], temp[2]});
        }
    }
}

std::vector<float> Bread::labToRgb(float L, float A, float B){

    float fy = (L + 16.0f) / 116.0f;
    float fx = A / 500.0f + fy;
    float fz = fy - B / 200.0f;
    float delta = 6.0f / 29.0f;

    float x = (fx > delta) ? std::pow(fx, 3.0f) : 3 * delta * delta * (fx - 4.0f / 29.0f);
    x *= 95.047f;
    float y = (fy > delta) ? std::pow(fy, 3.0f) : 3 * delta * delta * (fy - 4.0f / 29.0f);;
    y *= 100.f;
    float z = (fz > delta) ? std::pow(fz, 3.0f) : 3 * delta * delta * (fz - 4.0f / 29.0f);;
    z *= 108.883f;

    x /= 100.0f;
    y /= 100.0f;
    z /= 100.0f;

    float r = x *  3.2406f + y * -1.5372f + z * -0.4986f;
    float g = x * -0.9689f + y *  1.8758f + z *  0.0415f;
    float b = x *  0.0557f + y * -0.2040f + z *  1.0570f;

    r = (r > 0.0031308f) ? (1.055f * std::pow(r, 1 / 2.4f) - 0.055f) : (12.92f * r);
    g = (g > 0.0031308f) ? (1.055f * std::pow(g, 1 / 2.4f) - 0.055f) : (12.92f * g);
    b = (b > 0.0031308f) ? (1.055f * std::pow(b, 1 / 2.4f) - 0.055f) : (12.92f * b);

    r = std::clamp(r, 0.f, 1.f);
    g = std::clamp(g, 0.f, 1.f);
    b = std::clamp(b, 0.f, 1.f);

    return {r, g, b};
}

void Bread::saveMTL(){

    ofstream mtlFile("material.mtl"); // Creates or overwrites material.mtl

    if (!mtlFile) {
        throw runtime_error("Failed to create .mtl file!");
    }

    mtlFile << "newmtl crust\n";
    mtlFile << "Ka 1.000 1.000 1.000\n"; // ambient color
    mtlFile << "Kd 1.000 1.000 1.000\n"; // diffuse color
    mtlFile << "Ks 0.000 0.000 0.000\n"; // specular color
    mtlFile << "d 1.0\n"; // opacity
    mtlFile << "Ns 10.0\n"; // specular exponent
    mtlFile << "illum 2\n"; // lighting model (ours is diffuse and specular)
    mtlFile << "map_Kd crust_color.jpg\n"; // path to the texture image

    mtlFile.close();
}

void Bread::saveJPG(){

    // Maps distance to color:
    std::map<double, std::vector<double>> rgb_dict;
    for(int i = 0; i < rgb_colors.size(); i++){
        rgb_dict[rgb_colors[i][0]] = {rgb_colors[i][1], rgb_colors[i][2], rgb_colors[i][3]};
    }

    int width = 1;
    int height = rgb_dict.size();
    std::vector<std::vector<double>> rgb_values;

    for (const auto& pair : rgb_dict) {
        rgb_values.push_back(pair.second);
    }

    cv::Mat image(height, width, CV_8UC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            image.at<cv::Vec3b>(y, x)[2] = rgb_values[y][0]; // red
            image.at<cv::Vec3b>(y, x)[1] = rgb_values[y][1]; // green
            image.at<cv::Vec3b>(y, x)[0] = rgb_values[y][2]; // blue
        }
    }

    if (!cv::imwrite("crust_color.jpg", image)) {
        throw runtime_error("Could not save image\n");
    }
}
using namespace std;
using namespace Eigen;

// safe voxel lookup
inline bool getVoxel(const vector<bool>& voxels, int x, int y, int z, int dimX, int dimY, int dimZ) {
    if (x < 0 || y < 0 || z < 0 || x >= dimX || y >= dimY || z >= dimZ) return false;
    return voxels[x + dimX * (y + dimY * z)];
}

// midpoint interpolation
Vector3f interpolate(const Vector3f& p1, const Vector3f& p2) {
    return 0.5f * (p1 + p2);
}

// hashing for Vector3f
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

void Bread::marchingCubes(const vector<bool>& voxels, int dimX, int dimY, int dimZ,
                          vector<Vector3f>& outVertices, vector<Triangle>& outTriangles,
                          const int edgeTable[256], const int triTable[256][16]) {
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

void Bread::saveOBJ(const string& filename, const vector<Vector3f>& vertices, const vector<Triangle>& triangles) {

    ofstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Failed to open OBJ file for writing!");
    }

    file << "#\n";
    file << "# object crust\n";
    file << "#\n";
    file << "\n";

    //vertices
    for (const auto& v : vertices){
        file << "v " << v.x() << " " << v.y() << " " << v.z() << "\n";
    }
    file << "# " << vertices.size() << " vertices\n";
    file << "\n";

    //texture coords
    std::map<double, std::vector<double>> rgb_dict;
    for(int i = 0; i < rgb_colors.size(); i++){
        rgb_dict[rgb_colors[i][0]] = {rgb_colors[i][1], rgb_colors[i][2], rgb_colors[i][3]};
    }
    int num_coords = rgb_dict.size();

    for(int i = 0; i < num_coords; i++){
        file << "vt 0.000 " << (float)num_coords << " 0.000\n";
    }
    file << "# " << num_coords << " texture coords\n";
    file << "\n";

    file << "g crust\n";
    file << "usemtl crust\n";
    file << "s 0\n";

    std::vector<Vector3f> voxel_locs;
    for (int i = 0; i < m_voxels.size(); i++){
        int x, y, z;
        if(m_distance_voxels[i] <= crust_thickness){
            voxelToIndices(m_voxels[i], x, y, z);
            voxel_locs.push_back(Vector3f(x, y, z));
        }
    }

    std::map<int, int> vertex_to_voxel;
    for(int i = 0; i < outVertices.size(); i++){

        Vector3f closest = Vector3f(INFINITY, INFINITY, INFINITY);
        int num_closest = INFINITY;
        for(int j = 0; j < voxel_locs.size(); j++){

            double x_diff = abs(outVertices[i].x() - voxel_locs[j].x());
            double y_diff = abs(outVertices[i].y() - voxel_locs[j].y());
            double z_diff = abs(outVertices[i].z() - voxel_locs[j].z());

            double x_sim = abs(closest.x() - voxel_locs[j].x());
            double y_sim = abs(closest.y() - voxel_locs[j].y());
            double z_sim = abs(closest.z() - voxel_locs[j].z());

            if(x_diff < x_sim && y_diff < y_sim && z_diff < z_sim){
                closest = outVertices[i];
                num_closest = j;
            }
        }
        vertex_to_voxel[i] = num_closest;
    }

    for (const auto& t : triangles){

        int curr_voxel_0 = vertex_to_voxel[t.v0];
        int curr_voxel_1 = vertex_to_voxel[t.v1];
        int curr_voxel_2 = vertex_to_voxel[t.v2];

        int coord0 = floor(m_distance_voxels[curr_voxel_0]) / num_coords;
        coord0 = floor(coord0);
        int coord1 = floor(m_distance_voxels[curr_voxel_1]) / num_coords;
        coord1 = floor(coord1);
        int coord2 = floor(m_distance_voxels[curr_voxel_2]) / num_coords;
        coord2 = floor(coord2);

        file << "f "
             << (t.v0 + 1) << "/" << coord0 << " "
             << (t.v1 + 1) << "/" << coord1 << " "
             << (t.v2 + 1) << "/" << coord2 << " " << "\n";
    }
}
