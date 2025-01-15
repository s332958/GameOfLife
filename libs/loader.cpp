#include <fstream>
#include <iostream>
#include <iomanip>
#include <stdlib.h>

// Funzione principale per leggere il file
void readWorld(char *filename, int *dim_world, float **world, int **id_matrix) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file!" << std::endl;
        return;
    }
    int d;
    file >> d;

    *world = new float[d*d];
    *id_matrix = new int[d * d];

    for(int i=0;i<d*d;i++) {
        file>>(*world)[i];
        if((*world)[i]>0) (*id_matrix)[i]=-1;
        else (*id_matrix)[i] = 0;
    }
    *dim_world = d;

    file.close();
}

void readMatrix(char *filename, int *dim, float **matrix){
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file!" << std::endl;
        return;
    }
    int d;
    file >> d;

    *matrix = new float[d*d];

    for(int i=0;i<d*d;i++) {
        file>>(*matrix)[i];
    }
    *dim = d;

    file.close();
}

void printing_world(char* description, float *world, int *id_matrix, int dim_world){
    std::cout<<"START PRINTING "<<description<<": \n\n";
    for(int i=0; i<dim_world; i++){
        for(int j=0; j<dim_world; j++){
            std::cout<< std::setw(5) << world[i*dim_world + j] << " ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
    for(int i=0; i<dim_world; i++){
        for(int j=0; j<dim_world; j++){
            std::cout<< std::setw(5) << id_matrix[i*dim_world + j] << " ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}

void printing_matrix(char* description, float *matrix, int dim){
    std::cout<<"START PRINTING "<<description<<": \n\n";
    for(int i=0; i<dim; i++){
        for(int j=0; j<dim; j++){
            std::cout<< std::setw(5) << matrix[i*dim+ j] << " ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}