#include "bread.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "marching.h"
#include <omp.h>
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
    const std::string& filepath = "meshes-binvox/bread_128.binvox";

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

    // if (index != numVoxels) {
    //     cout << "not matched" << endl;
    // }

    file.close();

    // int dim_X = 10, dim_Y = 10, dim_Z = 10;
    // vector<bool> voxels(dim_X * dim_Y * dim_Z, false);

    // // Fill a 3x3x3 cube in the center
    // for (int z = 4; z < 7; ++z)
    //     for (int y = 4; y < 7; ++y)
    //         for (int x = 4; x < 7; ++x)
    //             voxels[toIndex(x, y, z, dim_X, dim_Y)] = true;

    // extractVoxelSurfaceToOBJ(voxels, dim_X, dim_Y, dim_Z, "output.obj");

    // PADDING
    // add padding around the edges to allow for rising
    addPadding(10);

    m_P.resize(m_voxels.size());
    std::fill(m_P.begin(), m_P.end(), 1);

    fillIn();

    // NAIVE
    // extractVoxelSurfaceToOBJ(m_voxels, dimX, dimY, dimZ, "bread-output.obj");

    // // MARCHING
    // vector<Eigen::Vector3f> vertices;
    // // vector<Eigen::Vector3f> normals;
    // vector<Triangle> triangles;

    // marchingCubes(m_voxels, dimX, dimY, dimZ, vertices, triangles, edgeTable, triangleTable);

    // saveOBJ("bread_mesh_128_deform.obj", vertices, triangles);

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
    generateBubbles(1, 9);

    initTemperatures();
    initBake();

    for (int i = 0; i < bakingIterations; i++) {
        bake();
        // temps are nan when in release mode but not in debug
    }

    for (int i = 0; i < m_temperatures.size(); i++) {
        cout << m_temperatures[i] - 273.15 << endl;
    }

    heatMap();

    cout << "done!" << endl;

    // std::vector<bool> voxelCopy = m_voxels;
    // // do cross section
    // for (int i = 0; i < m_voxels.size(); i++) {
    //     int x, y, z;
    //     voxelToIndices(i, x, y, z);
    //     // cout << "x: " << x << endl;
    //     // cout << "y: " << y << endl;
    //     // cout << "z: " << z << endl;
    //     if (y < dimY / 2) {
    //         // cout << "hi" << endl;
    //         // set to 0
    //         voxelCopy[i] = 0;
    //     }
    // }

    // writeBinvox("original-128-fix.binvox", dimX, dimY, dimZ, voxelCopy, translateX, translateY, translateZ, scale);

    // constructMockTemp();
    // generateGaussianFilter();
    // convolveGaussian();

    // // // std::vector<std::vector<float>> gradient = calcGradient(100);
    // // // std::cout << gradient[0][5] << std::endl;
    // // // std::cout << gradient[1][5] << std::endl;
    // // // std::cout << gradient[2][5] << std::endl;

    // m_gradVector = calcGradient(m_mock_temp);

    // warpBubbles(m_gradVector);
    // rise(m_gradVector);

    // for (int i = 0; i < m_voxels.size(); i++) {
    //     int x, y, z;
    //     voxelToIndices(i, x, y, z);
    //     // cout << "x: " << x << endl;
    //     // cout << "y: " << y << endl;
    //     // cout << "z: " << z << endl;
    //     if (y < dimY / 2) {
    //         // cout << "hi" << endl;
    //         // set to 0
    //         m_voxels[i] = 0;
    //     }
    // }

    // writeBinvox("128-rise-fix.binvox", dimX, dimY, dimZ, m_voxels, translateX, translateY, translateZ, scale);
}

void Bread::distanceVoxels() {
    m_distance_voxels.resize(m_voxels.size());

    #pragma omp parallel for
    for (int i = 0; i < m_voxels.size(); i++) {
        if (!m_voxels[i]) {
            m_distance_voxels[i] = 0.f;
        } else {
            int x, y, z;
            voxelToIndices(i, x, y, z);
            if (x == 0 || y == 0 || z == 0 || x == dimX - 1 || y == dimY - 1 || z == dimZ - 1) {
                m_distance_voxels[i] = 1.f;
            } else {
                float minDistance = dimX * dimY * dimZ;
                for (int j = 0; j < m_voxels.size(); j++) {
                    if (i != j && !m_voxels[j]) {
                        int x_prime, y_prime, z_prime;
                        voxelToIndices(j, x_prime, y_prime, z_prime);
                        float currDistance = sqrt(std::pow(x - x_prime, 2) + std::pow(y - y_prime, 2) + std::pow(z - z_prime, 2));
                        if (currDistance < minDistance) {
                            minDistance = currDistance;
                        }
                    }
                }
                if (x + 1 < minDistance) {
                    minDistance = x + 1;
                }
                if (y + 1 < minDistance) {
                    minDistance = y + 1;
                }
                if (y + 1 < minDistance) {
                    minDistance = z + 1;
                }
                if (dimX - x < minDistance) {
                    minDistance = dimX - x;
                }
                if (dimY - y < minDistance) {
                    minDistance = dimY - y;
                }
                if (dimZ - z < minDistance) {
                    minDistance = dimZ - z;
                }
                m_distance_voxels[i] = minDistance;
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
                                bool zz = m_voxels[index];
                                if (m_voxels[index] == 0) {
                                   // cout << "filling in" << endl;
                                }
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
void Bread::bake(){

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
}

void Bread::initBake() {
    // fill m_W
    m_W.assign(m_temperatures.size(), 0.4);

    // fill m_p
    m_p.reserve(m_temperatures.size());
    m_p.assign(m_temperatures.size(), 285.0);

}

void Bread::initTemperatures(){

    float largest = *std::max_element(m_distance_voxels.begin(), m_distance_voxels.end());
    // float largest = 12;
    cout << largest << endl;
    m_temperatures.resize(int(largest));
    m_temperatures.assign(int(largest), 298.0); //23 degrees celsius for every location
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
