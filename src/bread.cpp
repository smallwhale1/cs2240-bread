#include "bread.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <QString>
#include <QFile>
#include <QTextStream>
#include <QRegularExpression>
#include <QFileInfo>

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

    int dimX;
    int dimY;
    int dimZ;

    while (std::getline(file, line)) {
        if (line.substr(0, 3) == "dim") {
            sscanf(line.c_str(), "dim %d %d %d", &dimX, &dimY, &dimZ);
        } else if (line.substr(0, 10) == "translate ") {
            // handle translate
        } else if (line.substr(0, 6) == "scale ") {
            // handle scale
        } else if (line == "data") {
            // end of header
            break;
        }
    }

    // data
    // # voxels
    int numVoxels = dimX * dimY * dimZ;

    cout << "Num voxels: " << numVoxels << endl;

    std::vector<bool> voxels(numVoxels);

    // populate voxels
    char val;
    int index = 0;
    while (file.read(reinterpret_cast<char*>(&val), 1)) {
        for (int i = 0; i < 8; ++i) {
            if (index < numVoxels) {
                voxels[index++] = (val >> i) & 1;
            }
        }
    }

    // cout << voxels[0] << endl;

    file.close();

}
//iterates over every distance for each time step, stores results in temperatures vector
void bake(){

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
    float specific_heat = 3500.f; //amount of heat required to increase the temperature of a specific material by one degree

    float density = 284.f; //284 of initial condition of dough, TODO: changes by 170 + 284W for each time step



}

void calcHeatTranferCoeff(){



}
