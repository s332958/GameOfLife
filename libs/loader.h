#include <string>
#include <vector>
#include <sstream>

#ifndef UTILS
#define UTILS

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
        std::string worldName;
        std::vector<std::string> creatureListNames, creatureFilterListName;
        std::vector<Posizione> creturesPositions;

        SimulationSetup(std::string worldNameFile){
            worldName = worldNameFile;
            numberCreatures = 0;
        }

        void addCreatureName(std::string creatureName, int posx, int posy, std::string filter){
            creatureListNames.push_back(creatureName);
            creturesPositions.push_back(Posizione(posx,posy));
            creatureFilterListName.push_back(filter);
            numberCreatures++;
        }

        std::string toString(){
            std::ostringstream oss;
            oss << "World Name: " << worldName << "\n";
            oss << "Number of Creatures: " << numberCreatures << "\n";
            for (size_t i = 0; i < creatureListNames.size(); i++) {
                oss << "  Creature " << i + 1 << ": " << creatureListNames[i] 
                    << " (Filter: " << creatureFilterListName[i] 
                    << ", Position: [" << creturesPositions[i].getX() << ", " << creturesPositions[i].getY() << "])\n";
            }
            return oss.str();
        }

};

std::vector<SimulationSetup> readConfiguration(std::string fileName);

#endif
