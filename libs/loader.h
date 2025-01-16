#include <string>

void readWorld(const std::string& filename, int* dim, float** matrix, int** id_matrix);
void readMatrix(const std::string& filename, int* dim, float** matrix);
void printing_world(const std::string& description, float* world, int* id_matrix, int dim_world);
void printing_matrix(const std::string& description, float* matrix, int dim);
void clear_file(const std::string& filename);
void create_matrix(const std::string& filename, int dim, int value);
void save_matrix_to_file(const std::string& filename, int* matrix, int dim);
void save_matrix_to_file(const std::string& filename, float* matrix, int dim);
void save_matrix_to_file(std::ofstream& file, float* matrix, int dim);
void save_matrix_to_file(std::ofstream& file, int* matrix, int dim);