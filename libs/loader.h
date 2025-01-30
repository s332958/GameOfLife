#include <string>
#include <vector>

void readWorld(const std::string& filename, int* dim, float** matrix, int** id_matrix);
void readMatrix(const std::string& filename, int* dim, float** matrix);
void printing_world(const std::string& description, float* world, int* id_matrix, int dim_world);
void printing_matrix(const std::string& description, float* matrix, int dim);
void clear_file(const std::string& filename, bool debug=false);
void create_matrix(const std::string& filename, int dim, int value);
void save_matrix_to_file(const std::string& filename, int* matrix, int dim, bool debug=false);
void save_matrix_to_file(const std::string& filename, float* matrix, int dim, bool debug=false);
void save_matrix_to_file(std::ofstream& file, float* matrix, int dim, bool debug=false);
void save_matrix_to_file(std::ofstream& file, int* matrix, int dim, bool debug=false);

class Posizione {
public:
    int x;
    int y;

    // Costruttore
    Posizione(int x_val = 0, int y_val = 0) : x(x_val), y(y_val) {}

    // Metodo per ottenere la posizione x
    int getX() const { return x; }

    // Metodo per ottenere la posizione y
    int getY() const { return y; }

    // Metodo per settare la posizione x
    void setX(int x_val) { x = x_val; }

    // Metodo per settare la posizione y
    void setY(int y_val) { y = y_val; }
};

class SimulationSetup{

    public:
        int numberCreatures;
        std::string worldName, filterName;
        std::vector<std::string> creatureListNames;
        std::vector<Posizione> creturesPositions;

        SimulationSetup(std::string worldNameFile, std::string filterNameFile){
            worldName = worldNameFile;
            filterName = filterNameFile;
            numberCreatures = 0;
        }

        void addCreatureName(std::string creatureName, int posx, int posy){
            creatureListNames.push_back(creatureName);
            creturesPositions.push_back(Posizione(posx,posy));
            numberCreatures++;
        }

};

std::vector<SimulationSetup> readConfiguration(std::string fileName);
