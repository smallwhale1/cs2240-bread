#include "bread.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QFileInfo>
#include <cmath>

using namespace std;

Bread::Bread() {
}

void Bread::init() {
    // specify voxel filepath

    // absolute right now
    const std::string& filepath = "meshes-binvox/bun_48x23x48.binvox";

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

    fillIn();

    distanceVoxels();

    int x, y, z;
    voxelToIndices(200, x, y, z);
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
    std::cout << "z: " << z << std::endl;

    int i;
    indicesToVoxel(x, y, z, i);
    std::cout << "i: " << i << std::endl;

    generateSphere(0, 0, 0, 2);
    generateBubbles(1, 5);

    // do cross section
    for (int i = 0; i < m_voxels.size(); i++) {
        int x, y, z;
        voxelToIndices(i, x, y, z);
        // cout << "x: " << x << endl;
        // cout << "y: " << y << endl;
        // cout << "z: " << z << endl;
        if (x < 24) {
            // cout << "hi" << endl;
            // set to 0
            m_voxels[i] = 0;
        }
    }

    // writeBinvox("test.binvox", dimX, dimY, dimZ, m_voxels, translateX, translateY, translateZ, scale);


    initTemperatures();
    initW();

    for (int i = 0; i < bakingIterations; i++) {
        bake();
    }

    for (int i = 0; i < m_temperatures.size(); i++) {
        cout << m_temperatures[i] << endl;
    }

    cout << "done!" << endl;
}

void Bread::distanceVoxels() {
    m_distance_voxels.resize(m_voxels.size());
    for (int i = 0; i < m_voxels.size(); i++) {
        if (!m_voxels[i]) {
            m_distance_voxels[i] = 0.f;
        } else {
            int x, y, z;
            voxelToIndices(i, x, y, z);
            // cout << "x: " << x << endl;
            // cout << "y: " << y << endl;
            // cout << "z: " << z << endl;
            float minDistance = dimX * dimY * dimZ;
            for (int j = 0; j < m_voxels.size(); j++) {
                if (i != j && !m_voxels[j]) {
                    int x_prime, y_prime, z_prime;
                    voxelToIndices(j, x_prime, y_prime, z_prime);
                    float currDistance = sqrt(std::pow(x - x_prime, 2) + std::pow(y - y_prime, 2) + std::pow(z - z_prime, 2));
                    cout << currDistance << endl;
                    if (currDistance < minDistance) {
                        minDistance = currDistance;
                    }
                }
            }
            // cout << minDistance << endl;
            m_distance_voxels[i] = minDistance;
        }
    }
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
                if (distance <= radius) {
                    m_voxels[idx] = 0;
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
    int r = 64; // resolution of proving vol in each spatial coordinate
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
void Bread::bake(){

    //hr, hc

    float hc = 0.5f; //heat transfer coefficient for convection

    float sigma = 5.670374419e-8f;
    float temp_air; //temp in air
    float temp_surface; //temp at surface of dough
    float temp_radial = 283.15; //temp at heat source
    float emissivity_product = 0.9f;
    float emissivity_radial;

    float asp = 1.0f; //TODO: length of the sample
    float bsp = 1.0f; //TODO: width of the sample
    float lsp; //distance between radial source and sample, TODO: from distance mapping
    float a = asp / lsp;
    float b = bsp / lsp;
    float a1 = 1.0f + pow(a, 2);
    float b1 = 1.0f + pow(b, 2);

    //shape factor, compares shape of bread to shape of radial source
    float fsp = (2.0 / (M_PI * a * b)) * (
                    std::log( sqrt((a1 * b1) / (1.0f + pow(a, 2) + pow(b, 2)))) +
                    (a * sqrt(b1) * std::atan(a / sqrt(b1))) +
                    (b * sqrt(a1) * std::atan(b / sqrt(a1))) -
                    (a * std::atan(a)) - (b * std::atan(b))
                    );

    //heat transfer coefficient for radiation
    float hr = (sigma * (pow(temp_radial, 2) + pow(temp_surface, 2)) * (temp_radial + temp_surface)) /
               ((1.0f / emissivity_product) + (1.0f / emissivity_radial) - 2.0f + (1.0f / fsp));



    float k = 0.07; // thermal conductivity
    float lambda = 2.257f; //latent heat of evaporation/vaporization of water (like how much heat is needs for a phase change i think)
    float diffusivity = 1.35e-10f; //liquid water diffusivity (inversely proportoinal to viscosity, basically how easily it mixes with other stuff)
    float h_w = 1.f; // mass transfer coefficient water
    float h_v = 1.f; // mass transfer coefficient vapor
    float specific_heat = 3500.f; //amount of heat required to increase the temperature of a specific material by one degree

    float density = 284.f; //284 of initial condition of dough, TODO: changes by 170 + 284W for each time step

    float distance = 1.f;

    std::vector<float> dtdt(m_temperatures.size(), 0.f); //change in temperature over time
    std::vector<float> dtdx(m_temperatures.size(), 0.f); //change in temperature over distance
    std::vector<float> dtdx2(m_temperatures.size(), 0.f); //change in roc of temprature over distance

    std::vector<float> dWdt(m_temperatures.size(), 0.f); // change in water diffusion over t
    std::vector<float> dWdx(m_temperatures.size(), 0.f); // change in water diffusion over x
    std::vector<float> dWdx2(m_temperatures.size(), 0.f); // derivative of dWdx

    // fill in dWdx
    dWdx[0] = h_w * (m_W[0] - temp_air);
    dWdx[m_W.size() - 1] = 0.f;

    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            continue;
        }
        dWdx[x] = (m_W[x+1] - m_W[x-1]) / (distance * 2.f); // avg rate of change
    }

    // fill in dWdx2
    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            continue;
        }
        dWdx2[x] = (dWdx[x+1] - dWdx[x-1]) / (distance * 2.f);
    }

    // fill in dWdt
    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            continue;
        }
        dWdt[x] = diffusivity * dWdx2[x];
    }

    // timestep forward
    for (int x = 0; x < m_W.size(); x++) {
        if (x == 0 || x == m_W.size() - 1) {
            continue;
        }
        // explicit euler for now
        m_W[x] += timestep * dWdt[x];
    }

    // LANA maybe check this: fill in edge cases for dWdt
    m_W[0] = m_W[1] - dWdx[0];
    m_W[m_W.size() - 1] = m_W[m_W.size() - 2];

    //fill up dtdx
    for(int x = 0; x < m_temperatures.size(); x++){

        if(x == 0){ //outside edge of the bread
            dtdx[x] = 0.f;

        } else if(x == m_temperatures.size() - 1){ //inside edge of the bread
            dtdx[x] = (hr * (temp_radial - temp_surface)) + (hc * (temp_air - temp_surface)) - (lambda * density * diffusivity * (dWdx[0]));

        } else { //every other internal point in the bread
            dtdx[x] = (m_temperatures[x - 1] + m_temperatures[x + 1]) / (distance * 2.f);
        }
    }

    for(int x = 0; x < dtdx2.size(); x++){

        if(x == 0){ //outside edge of the bread
            dtdx2[x] = (dtdx[1] - dtdx[0]) / distance;

        } else if(x == m_temperatures.size() - 1){ //inside edge of the bread
            dtdx2[x] = (dtdx[x] - dtdx[x - 1]) / distance;

        } else { //every other internal point in the bread
            dtdx2[x] = (dtdx[x - 1] + dtdx[x + 1]) / (distance * 2.f);
        }
    }

    // fill up dtdt
    for (int x = 0; x < dtdt.size(); x++) {
        float new_p = 170.f + (284.f * m_W[x]);
        float dpdt = (new_p - m_p[x]) / distance;
        m_p[x] = new_p;
        dtdt[x] = (k * dtdx2[x]) / (new_p * specific_heat) + (lambda * dWdt[x]) / specific_heat + (lambda * m_W[x] * dpdt) / (new_p * specific_heat);
    }


    // update m_temperatures
    for (int x = 0; x < m_temperatures.size(); x++) {
        m_temperatures[x] += timestep * dtdt[x];
    }

    // update m_p
    // for (int x = 0; x < m_p.size(); x++) {
    //     m_p[x] = 170.f + 284.f * m_W[x];
    // }
}

void Bread::initW() {
    // fill m_W
    // m_W.resize(m_temperatures.size());
    m_W.assign(m_temperatures.size(), 0.4f);

    // fill m_p
    m_p.reserve(m_temperatures.size());
    m_p.assign(m_temperatures.size(), 285.f);

}

void Bread::initTemperatures(){

    float largest = *std::max_element(m_distance_voxels.begin(), m_distance_voxels.end());
    cout << largest << endl;
    m_temperatures.resize(largest / 2);
    m_temperatures.assign(largest / 2, 25.0f); //23 degrees celsius for every location
}

void calcHeatTranferCoeff(){



}
