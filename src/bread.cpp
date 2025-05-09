#include "bread.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
// #include <omp.h>
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
    // specify voxel filepath

    // absolute right now
    const std::string& filepath = "meshes-binvox/bread_256.binvox";

    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open file");
    }

    // load in voxel into bool vector

    std::string line;
    std::getline(file, line);
    if (line != "#binvox 1") {
        throw std::runtime_error("Invalid header");
    }

    while (std::getline(file, line)) {
        if (line.substr(0, 3) == "dim") {
            sscanf(line.c_str(), "dim %d %d %d", &dimX, &dimY, &dimZ);
        } else if (line.substr(0, 10) == "translate ") {
            // handle translate
            sscanf(line.c_str(), "translate %f %f %f", &translateX, &translateY, &translateZ);
        } else if (line.substr(0, 6) == "scale ") {
            // handle scale
            sscanf(line.c_str(), "scale %f", &scale);
        } else if (line == "data") {
            // end of header
            break;
        }
    }

    // data
    // # voxels
    int numVoxels = dimX * dimY * dimZ;

    m_voxels.resize(numVoxels);
    m_P.resize(numVoxels);
    std::fill(m_P.begin(), m_P.end(), 1);
    // std::vector<bool> voxels(numVoxels);

    // parse data
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

    // int dim_X = 10, dim_Y = 10, dim_Z = 10;
    // vector<bool> voxels(dim_X * dim_Y * dim_Z, false);

    // // Fill a 3x3x3 cube in the center
    // for (int z = 4; z < 7; ++z)
    //     for (int y = 4; y < 7; ++y)
    //         for (int x = 4; x < 7; ++x)
    //             voxels[toIndex(x, y, z, dim_X, dim_Y)] = true;

    // extractVoxelSurfaceToOBJ(voxels, dim_X, dim_Y, dim_Z, "output.obj");

    // add padding around the edges to allow for rising
    addPadding(5);

    m_P.resize(m_voxels.size());
    std::fill(m_P.begin(), m_P.end(), 1);

    fillIn();

    // NAIVE
    // extractVoxelSurfaceToOBJ(m_voxels, dimX, dimY, dimZ, "bread-output.obj");

    // vector<Eigen::Vector3f> vertices;
    // // vector<Eigen::Vector3f> normals;
    // vector<Triangle> triangles;

    // marchingCubes(m_voxels, dimX, dimY, dimZ, vertices, triangles, edgeTable, triangleTable);

    // saveOBJ("bread_mesh_128.obj", vertices, triangles);

    // int x, y, z;
    // voxelToIndices(200, x, y, z);
    // std::cout << "x: " << x << std::endl;
    // std::cout << "y: " << y << std::endl;
    // std::cout << "z: " << z << std::endl;

    // int i;
    // indicesToVoxel(x, y, z, i);
    // std::cout << "i: " << i << std::endl;

    // BREAD LOGIC

    distanceVoxels();
    generateSphere(0, 0, 0, 2);
    generateBubbles(1, 7);

    std::vector<bool> voxelCopy = m_voxels;
    // do cross section
    for (int i = 0; i < m_voxels.size(); i++) {
        int x, y, z;
        voxelToIndices(i, x, y, z);
        // cout << "x: " << x << endl;
        // cout << "y: " << y << endl;
        // cout << "z: " << z << endl;
        if (y < dimY / 2) {
            // cout << "hi" << endl;
            // set to 0
            voxelCopy[i] = 0;
        }
    }

    writeBinvox("test-original-128.binvox", dimX, dimY, dimZ, voxelCopy, translateX, translateY, translateZ, scale);

    initTemperatures();
    initBake();

    for (int i = 0; i < bakingIterations; i++) {
        bake(i);
    }

    for (int i = 0; i < m_temperatures.size(); i++) {
        cout << m_temperatures[i] - 273.15 << endl;
    }

//    heatMap();

    constructTemp();
    // generateGaussianFilter();
    // convolveGaussian();

    // // std::vector<std::vector<float>> gradient = calcGradient(100);
    // // std::cout << gradient[0][5] << std::endl;
    // // std::cout << gradient[1][5] << std::endl;
    // // std::cout << gradient[2][5] << std::endl;

    m_gradVector = calcGradient(m_temp);

    warpBubbles(m_gradVector);
    rise(m_gradVector);

    for (int i = 0; i < m_voxels.size(); i++) {
        int x, y, z;
        voxelToIndices(i, x, y, z);
        // cout << "x: " << x << endl;
        // cout << "y: " << y << endl;
        // cout << "z: " << z << endl;
        if (y < dimY / 2) {
            // cout << "hi" << endl;
            // set to 0
            m_voxels[i] = 0;
        }
    }

    writeBinvox("test-128-rise.binvox", dimX, dimY, dimZ, m_voxels, translateX, translateY, translateZ, scale);

    saveJPG();

    cout << "done!" << endl;
}


// void Bread::distanceVoxels() {
//     m_distance_voxels.resize(m_voxels.size());

//     #pragma omp parallel for
//     for (int i = 0; i < m_voxels.size(); i++) {
//         if (!m_voxels[i]) {
//             m_distance_voxels[i] = 0.f;
//         } else {
//             int x, y, z;
//             voxelToIndices(i, x, y, z);
//             if (x == 0 || y == 0 || z == 0 || x == dimX - 1 || y == dimY - 1 || z == dimZ - 1) {
//                 m_distance_voxels[i] = 1.f;
//             } else {
//                 float minDistance = dimX * dimY * dimZ;
//                 for (int j = 0; j < m_voxels.size(); j++) {
//                     if (i != j && !m_voxels[j]) {
//                         int x_prime, y_prime, z_prime;
//                         voxelToIndices(j, x_prime, y_prime, z_prime);
//                         float currDistance = sqrt(std::pow(x - x_prime, 2) + std::pow(y - y_prime, 2) + std::pow(z - z_prime, 2));
//                         if (currDistance < minDistance) {
//                             minDistance = currDistance;
//                         }
//                     }
//                 }
//                 if (x + 1 < minDistance) {
//                     minDistance = x + 1;
//                 }
//                 if (y + 1 < minDistance) {
//                     minDistance = y + 1;
//                 }
//                 if (y + 1 < minDistance) {
//                     minDistance = z + 1;
//                 }
//                 if (dimX - x < minDistance) {
//                     minDistance = dimX - x;
//                 }
//                 if (dimY - y < minDistance) {
//                     minDistance = dimY - y;
//                 }
//                 if (dimZ - z < minDistance) {
//                     minDistance = dimZ - z;
//                 }
//                 m_distance_voxels[i] = minDistance;
//             }

//         }
//     }
// }


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
                for (int dx = -1; dx <= 1 && !isSurface; ++dx) {
                    for (int dy = -1; dy <= 1 && !isSurface; ++dy) {
                        for (int dz = -1; dz <= 1 && !isSurface; ++dz) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            int nx = x + dx;
                            int ny = y + dy;
                            int nz = z + dz;
                            if (nx < 0 || ny < 0 || nz < 0 || nx >= dimX || ny >= dimY || nz >= dimZ)
                                continue;
                            int nIdx;
                            indicesToVoxel(nx, ny, nz, nIdx);
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
            int nx = x + dx[i];
            int ny = y + dy[i];
            int nz = z + dz[i];
            if (nx < 0 || ny < 0 || nz < 0 || nx >= dimX || ny >= dimY || nz >= dimZ)
                continue;
            int nIdx;
            indicesToVoxel(nx, ny, nz, nIdx);
            if (m_voxels[nIdx] && m_distance_voxels[nIdx] > currDist + 1) {
                m_distance_voxels[nIdx] = currDist + 1;
                q.emplace(nx, ny, nz);
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
    // x = index % (dimX * dimZ);
    // index -= x * dimX * dimZ;
    // z = index % dimZ;
    // y = index - z * dimZ;

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
    worldX = (x + 0.5f) / dimX;
    worldY = (y + 0.5f) / dimY;
    worldZ = (z + 0.5f) / dimZ;
    worldX = scale * worldX + translateX;
    worldY = scale * worldY + translateY;
    worldZ = scale * worldZ + translateZ;
}

bool Bread::voxelAt(int x, int y, int z) {
    return m_voxels[x * dimX * dimZ + z * dimZ + y];
}

void Bread::generateSphere(int x, int y, int z, int radius) {
    // get bounding box of size 2R + 1 in each dimension and store in vector
    // iterate through vector and check distance; if distance is < radius, set voxel == 0
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

                // bounds checking. TODO check: >= or > dimD depends on how dims are parsed. TODO: verify
                if (x_prime < 0 || x_prime >= dimX || y_prime < 0 || y_prime >= dimY || z_prime < 0 || z_prime >= dimZ) {
                    continue;
                }

                indicesToVoxel(x_prime, y_prime, z_prime, idx);

                // sanity bounds check. TODO: verify
                if (idx < 0 || idx > m_voxels.size()) {
                    continue;
                }

                float distance = sqrt(std::pow(x - x_prime, 2) + std::pow(y - y_prime, 2) + std::pow(z - z_prime, 2));
                // m_voxels[idx] = 1;

                // std::cout << distance << std::endl; // uncomment for checking that distance calculations are correct
                // add some padding around the crust
                if (distance <= radius && m_distance_voxels[idx] >= 3) {
                    m_voxels[idx] = 0;
                    // modify P
                    m_P[idx] = std::max(m_P[idx], radius);
                    count++;
                } else {
                    // m_voxels[idx] = 1; // TODO: prob can get rid of this since we want to maintain original mesh binary status; this is just for initial checking.
                }
            }
        }
    }

    // std::cout << "count: " << count << std::endl; // uncomment to check that we're subtracting the correct number of voxels for a given sphere
}

// use min of 2 and max of 20 for baguette setting
void Bread::generateBubbles(int minRadius, int maxRadius) {
    int radius = minRadius;

    // see page 9 for some constants. currently using baguette settings
    int r = 128; // resolution of proving vol in each spatial coordinate
    float k = 0.07 * pow(r, 3) * 0.05; // the amount of actual spheres at each radius
    float d = 2.78; // fractal exponent for likelihood of spheres given radii
    while (radius <= maxRadius) {
        // subtract spheres of minRadius
        int numSpheres = k / pow(radius, d);
        std::cout << "numSpheres: " << numSpheres << std::endl;
        for (int i = 0; i < numSpheres; i++) {
            // generate random coord inside dims
            int x = arc4random_uniform(dimX);
            int y = arc4random_uniform(dimY);
            int z = arc4random_uniform(dimZ);
            generateSphere(x, y, z, radius);
        }
        // increment
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
//iterates over every distance for each time step, stores results in temperatures vector
void Bread::bake(int time){

    //hr, hc

    double hc = 0.5; //heat transfer coefficient for convection

    double sigma = 5.670374419e-8;
    double temp_air = 483.150; //temp in air
    double temp_surface = m_temperatures[0]; //starting temp at the surface of the dough
    double temp_radial = 2.0 * temp_air; //temp at heat source
    double emissivity_product = 0.9;
    double emissivity_radial = 0.5; //tbh not sure but it's based on the this website and the common wire heating element in an oven of a nickel-copper mix
                                    //https://www.flukeprocessinstruments.com/en-us/service-and-support/knowledge-center/infrared-technology/emissivity-metals

    double asp = 0.1; //length of the sample
    double bsp = 0.2; //width of the sample
    double lsp = 0.5; //distance between radial source and sample, TODO: from distance mapping
    double a = asp / lsp;
    double b = bsp / lsp;
    double a1 = 1.0 + pow(a, 2);
    double b1 = 1.0 + pow(b, 2);

    //shape factor, compares shape of bread to shape of radial source
    double fsp = (2.0 / (M_PI * a * b)) * (
                    std::log( sqrt((a1 * b1) / (1.0 + pow(a, 2) + pow(b, 2)))) +
                    (a * sqrt(b1) * std::atan(a / sqrt(b1))) +
                    (b * sqrt(a1) * std::atan(b / sqrt(a1))) -
                    (a * std::atan(a)) - (b * std::atan(b))
                    );

    //heat transfer coefficient for radiation
    double hr = (sigma * (pow(temp_radial, 2) + pow(temp_surface, 2)) * (temp_radial + temp_surface)) /
               ((1.0 / emissivity_product) + (1.0 / emissivity_radial) - 2.0 + (1.0 / fsp));

    double k = 0.5; // thermal conductivity
    double lambda = 2.257; //latent heat of evaporation/vaporization of water (like how much heat is needs for a phase change i think)
    double diffusivity = 1.35e-10; //liquid water diffusivity (inversely proportoinal to viscosity, basically how easily it mixes with other stuff)
    double specific_heat = 3500.0; //amount of heat required to increase the temperature of a specific material by one degree

    double distance = 0.005;

    double w_air = 0.0; // paper says this is 0 but I expect that would change over time??

    std::vector<double> dtdt(m_temperatures.size(), 0.0); //change in temperature over time
    std::vector<double> dtdx(m_temperatures.size(), 0.0); //change in temperature over distance
    std::vector<double> dtdx2(m_temperatures.size(), 0.0); //change in roc of temprature over distance

    std::vector<double> dWdt(m_temperatures.size(), 0.0); // change in water diffusion over t
    std::vector<double> dWdx(m_temperatures.size(), 0.0); // change in water diffusion over x
    std::vector<double> dWdx2(m_temperatures.size(), 0.0); // derivative of dWdx

    // fill in dWdx
    double h_w = 0.00140 * m_temperatures[0] + 0.27 * m_W[0] - 0.0004 * m_temperatures[0] * m_W[0] - 0.77 * m_W[0] * m_W[0];
    dWdx[0] = h_w * (m_W[0] - w_air);
    dWdx[m_W.size() - 1] = 0.f;

    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            continue;
        }
        dWdx[x] = (m_W[x+1] - m_W[x-1]) / (distance * 2.0); // avg rate of change
    }

    // fill in dWdx2
    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            if(x == 0){ // outside edge of the bread
                dWdx2[x] = (dWdx[1] - dWdx[0]) / distance;

            } else if(x == m_temperatures.size() - 1){ // inside edge of the bread
                dWdx2[x] = (dWdx[x] - dWdx[x - 1]) / distance;
            }
        } else {
            dWdx2[x] = (dWdx[x+1] - dWdx[x-1]) / (distance * 2.0);
        }
    }

    // fill in dWdt
    for (int x = 0; x < m_W.size(); x++) {
        dWdt[x] = diffusivity * dWdx2[x];
    }

    // timestep forward
    for (int x = 0; x < m_W.size(); x++) {
        // explicit euler for now
        m_W[x] += timestep * dWdt[x];
    }

    m_W[0] = m_W[1] - distance * dWdx[0];
    m_W[m_W.size() - 1] = m_W[m_W.size() - 2];

    //fill up dtdx
    for(int x = 0; x < m_temperatures.size(); x++){
        if(x == m_temperatures.size() - 1){ //inside edge of the bread
            dtdx[x] = 0.0;
        } else if(x == 0){ //outside edge of the bread
            dtdx[x] = ((hr * (temp_radial - temp_surface)) + (hc * (temp_air - temp_surface)) - (lambda * m_p[x] * diffusivity * (dWdx[0]))) / (-k);
        } else { //every other internal point in the bread
            dtdx[x] = (m_temperatures[x + 1] - m_temperatures[x - 1]) / (distance * 2.0);
        }
    }

    for(int x = 0; x < dtdx2.size(); x++){
        if(x == 0){ //outside edge of the bread
            dtdx2[x] = (dtdx[1] - dtdx[0]) / distance;
        } else if(x == m_temperatures.size() - 1){ //inside edge of the bread
            dtdx2[x] = (dtdx[x] - dtdx[x - 1]) / distance;
        } else { //every other internal point in the bread
            dtdx2[x] = (dtdx[x + 1] - dtdx[x - 1]) / (distance * 2.0);
        }
    }

    // fill up dtdt
    for (int x = 0; x < dtdt.size(); x++) {
        double new_p = 170.0 + (284.0 * m_W[x]);
        double dpdt = (new_p - m_p[x]) / distance;
        m_p[x] = new_p;
        dtdt[x] = (k * dtdx2[x]) / (new_p * specific_heat) + (lambda * dWdt[x]) / specific_heat + (lambda * m_W[x] * dpdt) / (new_p * specific_heat);
    }

    // update m_temperatures
    for (int x = 0; x < m_temperatures.size(); x++) {
        m_temperatures[x] += timestep * dtdt[x];
    }

    //create crust
    createCrust(time, dWdt);
}

void Bread::initBake() {
    // fill m_W
    m_W.assign(m_temperatures.size(), 0.4);

    // fill m_p
    m_p.reserve(m_temperatures.size());
    m_p.assign(m_temperatures.size(), 285.0);

    // fill m_L
    m_L.assign(m_distance_voxels.size(), 70.f);

}

void Bread::initTemperatures(){

    float largest = *std::max_element(m_distance_voxels.begin(), m_distance_voxels.end());
    // float largest = 12;
    cout << largest << endl;
    m_temperatures.resize(int(largest));
    m_temperatures.assign(int(largest), 298.0); //23 degrees celsius for every location

    std::vector<float> m_L(m_temperatures.size(), 90.f);
    crust_time = 0.f;
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
                cout << idx << endl;
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

    // Normalize data and fill matrix
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            mat.at<float>(i, j) = data[i][j];

    cv::Mat normalized;
    cv::normalize(mat, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(normalized, CV_8U);

    cv::Mat colorMap;
    cv::applyColorMap(normalized, colorMap, cv::COLORMAP_JET);  // Choose any OpenCV colormap
    cv::imshow("Heatmap", colorMap);
    cv::waitKey(0);
}

void Bread::createCrust(int time, std::vector<double> dWdt){

    //has to change based on time step, relationship between num voxels of mesh and time step for thickness, certain percentage of crust based on timestep
    double crust_thickness = dimX * time * (0.03125 / bakingIterations); //not sure which dim this should be or if it should be an average / different for all dims
    //ok so this is based on me estimating one of the images that shows the crust thickness is 3.125% of the number of voxels when fully baked

    // a* [4.5e1.5] and b* [22.6e45.9]

    rgb_colors.clear();
        //channel 2 is positoin a between red and green (-120-+120);
        //chnanel 3 is position b between yellow and blue (-120-+120)
    bool addToCrust = true;

    // std::cout << "adding crust" << std::endl;
    for(int i = 0; i < m_distance_voxels.size(); i++){

        // std::cout << "distance i: " << m_distance_voxels[i] << ", crust thickness: " << crust_thickness << ", temp at this distance: " << m_temperatures[std::floor((m_distance_voxels[i]))] << ", dwdt at i: " << dWdt[i] << std::endl;

        //voxel that is within crust distance, has a temp > 120C/393.15K, and has water activity < 0.6 //increasing temperature decreases water activity
        if(m_distance_voxels[i] < crust_thickness && m_distance_voxels[i] != 0 && m_temperatures[std::floor((m_distance_voxels[i]))] > 350.f && m_W[std::floor((m_distance_voxels[i]))] < 0.6f){

            // cout << m_distance_voxels[i] << endl;
            if (addToCrust) {
                crust_time ++;
                addToCrust = false;
            }
            //how long have we been updating the crust color, hopefuly goes for about 20 iterations

            double t1 = std::pow(7.923310f, 6.f);
            float water_activity = m_W[m_distance_voxels[i]];
            double t2 = (std::pow(2.739710f, 6.f) / water_activity);
            double temp1 = t1 + t2; //water_acticity ranges from 0.1 to 0.6
            double temp2 = -1 * ((std::pow(8.701510, 3) + (49.4738 / water_activity)) / (m_temperatures[0])); //temp at this voxel at this time
            double k = temp1 * std::pow(M_E, temp2);
            k -= 34000;
            k /= 15000;
            m_L[i] += -k * m_L[i] * timestep / 60.f; //decrements

            float a = -4.5f + (.75 * crust_time);
            float b = 22.6 + (2.9f * crust_time);

            // std::cout << k << std::endl;
            // std::cout << m_distance_voxels[i]<< " " << m_L[i] << " " << a << " " << b << std::endl;
            std::vector<float> temp = labToRgb({m_L[i], a, b});
            rgb_colors.push_back({(float)m_distance_voxels[i], temp[0], temp[1], temp[2]});
        }

    }

    // std::cout << "color size: " << rgb_colors.size() << std::endl;
    // for(int i = 0; i < rgb_colors.size(); i++){
        // std::cout << "L: " << rgb_colors[i][0] << ", a: " << rgb_colors[i][1] << ", b: " << rgb_colors[i][2] << std::endl;
    // }


}

std::vector<float> Bread::labToRgb(std::vector<float> color){

    float L = color[0];
    float A = color[1];
    float B = color[2];

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
    g = (g > 0.0031308f) ? (1.055f * std::pow(g, 1 / 2.4f) - 0.055f) : (12.92f * g); //r?
    b = (b > 0.0031308f) ? (1.055f * std::pow(b, 1 / 2.4f) - 0.055f) : (12.92f * b); //r?

    r = std::clamp(r, 0.f, 1.f);
    g = std::clamp(g, 0.f, 1.f);
    b = std::clamp(b, 0.f, 1.f);

    return {r, g, b};

}



void Bread::saveMTL(){

    ofstream mtlFile("material.mtl"); // Creates or overwrites "material.mtl"

    if (!mtlFile) {
        throw runtime_error("Failed to create .mtl file!");
    }

    mtlFile << "newmtl crust\n";
    mtlFile << "Ka 1.000 1.000 1.000\n"; //ambient color
    mtlFile << "Kd 1.000 1.000 1.000\n"; //diffuse color (used if no texture is applied)
    mtlFile << "Ks 0.000 0.000 0.000\n"; //specular color
    mtlFile << "d 1.0\n"; //opacity (1 = fully opaque)
    mtlFile << "Ns 10.0\n"; //specular exponent
    mtlFile << "illum 2\n"; //lighting model (diffuse + specular)
    mtlFile << "map_Kd crust_color.jpg\n"; //path to the texture image

    mtlFile.close();
}

void Bread::saveJPG(){

    // maps distance to color
    std::map<double, std::vector<double>> rgb_dict;
    for(int i = 0; i < rgb_colors.size(); i++){
        rgb_dict[rgb_colors[i][0]] = {rgb_colors[i][1], rgb_colors[i][2], rgb_colors[i][3]};
    }

    int width = rgb_dict.size();
    int height = rgb_dict.size();
    std::vector<std::vector<double>> rgb_values;

    for (const auto& pair : rgb_dict) {
        rgb_values.push_back(pair.second);
    }

    // Create a 3-channel (BGR) image with 8-bit depth
    cv::Mat image(height, width, CV_8UC3);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {

            image.at<cv::Vec3b>(y, x)[2] = rgb_values[y][0] * 255; //red
            image.at<cv::Vec3b>(y, x)[1] = rgb_values[y][1] * 255; //green
            image.at<cv::Vec3b>(y, x)[0] = rgb_values[y][2] * 255; //blue

            cout << rgb_values[y][0] << " " << rgb_values[y][1] << " " << rgb_values[y][2] << endl;

        }
    }

    // Save the image as JPEG
    if (!cv::imwrite("crust_color.jpg", image)) {
        throw runtime_error("Could not save image\n");
    }
}

