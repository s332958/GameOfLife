#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include "loader.h"

// Funzione per leggere una matrice da file e calcolare id_matrix
void readWorld(const std::string& filename, int* dim_world, float** world, int** id_matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file!" << std::endl;
        return;
    }
    int d;
    file >> d;
    *world = (float*)malloc(d * d * sizeof(float));
    *id_matrix = (int*)malloc(d * d * sizeof(int));
    
    for (int i = 0; i < d * d; i++) {
        file >> (*world)[i];
        (*id_matrix)[i] = ((*world)[i] > 0) ? -1 : 0;
    }
    *dim_world = d;
}

// Funzione per leggere una matrice da file
void readMatrix(const std::string& filename, int* dim, float** matrix) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file!" << std::endl;
        return;
    }
    int d;
    file >> d;
    *matrix = (float*)malloc(d * d * sizeof(float));
    
    for (int i = 0; i < d * d; i++) {
        file >> (*matrix)[i];
    }
    *dim = d;
}

// Funzione per stampare la matrice world (DEBUG)
void printing_world(const std::string& description, float* world, int* id_matrix, int dim_world) {
    std::cout << "START PRINTING " << description << ":\n\n";
    for (int i = 0; i < dim_world; i++) {
        for (int j = 0; j < dim_world; j++) {
            std::cout << std::setw(5) << world[i * dim_world + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
    for (int i = 0; i < dim_world; i++) {
        for (int j = 0; j < dim_world; j++) {
            std::cout << std::setw(5) << id_matrix[i * dim_world + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Funzione per stampare una matrice generica (DEBUG)
void printing_matrix(const std::string& description, float* matrix, int dim) {
    std::cout << "START PRINTING " << description << ":\n\n";
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            std::cout << std::setw(5) << matrix[i * dim + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// Funzione per svuotare un file
void clear_file(const std::string& filename, bool debug){
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file!" << std::endl;
        return;
    }
    if(debug) std::cout << "Il file " << filename << " è stato svuotato correttamente.\n";
}

// Funzione per creare e scrivere una matrice su file
void create_matrix(const std::string& filename, int dim, int value) {
    std::ofstream file(filename, std::ios::out);
    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file!" << std::endl;
        return;
    }
    file << dim << "\n";
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            file << value << " ";
        }
        file << "\n";
    }
}


// Funzione per salvare un array di matrici su un file già aperto
void save_matrices_to_file(std::ofstream& file, unsigned char* matrix, int num_matrix, int dim_matrix, bool debug) {
    if (!file.is_open()) {
        std::cerr << "Errore: il file non è aperto!" << std::endl;
        return;
    }
    
    // Calcola il numero di matrici
    
    for (int m = 0; m < num_matrix; m++) {
        for (int i = 0; i < dim_matrix; i++) {
            for (int j = 0; j < dim_matrix; j++) {
                file << (int)matrix[m * dim_matrix * dim_matrix + i * dim_matrix + j] << " ";
            }
            file << "\n";
        }
        file << "\n"; // Separazione tra matricistatic_cast<int>()
    }
    
    std::cout << num_matrix << " matrici salvate correttamente nel file.\n";
}



/*

// Funzione per salvare una matrice su file
void save_matrix_to_file(const std::string& filename, float* matrix, int dim, bool debug) {
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Errore nella creazione del file!" << std::endl;
        return;
    }
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            file << matrix[i * dim + j] << " ";
        }
        file << "\n";
    }
    file << "\n";
    if(debug) std::cout << "Matrice salvata correttamente in " << filename << "\n";
}

void save_matrix_to_file(const std::string& filename, int* matrix, int dim, bool debug){
    std::ofstream file(filename, std::ios::app);
    if (!file.is_open()) {
        std::cerr << "Errore nella creazione del file!" << std::endl;
        return;
    }
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            file << matrix[i * dim + j] << " ";
        }
        file << "\n";
    }
    file << "\n";
    if(debug) std::cout << "Matrice salvata correttamente in " << filename << "\n";
}

// Funzione per salvare una matrice di tipo float in un file già aperto
void save_matrix_to_file(std::ofstream& file, float* matrix, int dim, bool debug ) {
    if (!file.is_open()) {
        std::cerr << "Errore: il file non è aperto!" << std::endl;
        return;
    }
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            file << matrix[i * dim + j] << " ";
        }
        file << "\n";
    }
    file << "\n";
    if(debug) std::cout << "Matrice salvata correttamente nel file.\n";
}

// Funzione per salvare una matrice di tipo int in un file già aperto
void save_matrix_to_file(std::ofstream& file, int* matrix, int dim, bool debug ) {
    if (!file.is_open()) {
        std::cerr << "Errore: il file non è aperto!" << std::endl;
        return;
    }
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            file << matrix[i * dim + j] << " ";
        }
        file << "\n";
    }
    file << "\n";
    if(debug) std::cout << "Matrice salvata correttamente nel file.\n";
}
*/

//funzione per la lettura delle configurazioni del programma
std::vector<SimulationSetup> readConfiguration(std::string fileName){

    std::vector<SimulationSetup> risultato;
    std::ifstream file(fileName);
    if (!file.is_open()) {
        std::cerr << "Errore nella lettura del file di configurazione!" << std::endl;
        return risultato;
    }
    
    while(!file.eof()){
        if(file){
            int numberCreatures;
            std::string worldName, filterName;
            file >> worldName;
            file >> numberCreatures;
            SimulationSetup sim(worldName);
            for(int i=0;i<numberCreatures;i++){
                int posX, posY;
                std::string creatureName;
                file >> creatureName;
                file >> posX;
                file >> posY;
                file >> filterName;
                sim.addCreatureName(creatureName,posX,posY,filterName);
            }
            risultato.push_back(sim);
        }
    }
    
    file.close();
    return risultato;

}