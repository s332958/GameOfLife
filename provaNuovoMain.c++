#include <time.h>
#include <GLFW/glfw3.h>

GLFWwindow* window;
GLuint textureID;

const int MAX_CREATURE = 64;

float colori[MAX_CREATURE][3];


int main(int argc, char* argv[]) {
    clock_t start = clock();  // Start time

    bool render = false;
    int N_EPHOCS = 1;
    int N_STEPS = 100;
    
    int const n_layer = 4;
    int v[n_layer] = {162,16,16,10};
    size_t dim_free = 1024*1024;
    int dim_world = 1024;
    int n_creature = 10;
    int const MAX_WORKSPACE = 10;
    int const EVAL_TYPE = 1;
    float *weight_model = nullptr;
    float *bias_model = nullptr;

    // Controlla se c'è almeno un argomento
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-render") {
            render = true;
        }
        if(arg == "-n_e"){
            N_EPHOCS = std::atoi(argv[i+1]);
        }
        if(arg == "-n_s"){
            N_STEPS = std::atoi(argv[i+1]);
        }
    }


    for (int i = 1; i <= 20; i++) {
        colori[i][0] = static_cast<float>(rand() % 256); // R
        colori[i][1] = static_cast<float>(rand() % 256); // G
        colori[i][2] = static_cast<float>(rand() % 256); // B
    }
    
    if(render){
        if (!glfwInit()) {
            std::cerr << "Errore nell'inizializzazione di GLFW" << std::endl;
            return -1;
        }
        window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Image Rendering", NULL, NULL);
    
        if (!window) {
            std::cerr << "Errore nella creazione della finestra" << std::endl;
            glfwTerminate();
            return -1;
        }
    
        glfwMakeContextCurrent(window);
        // Inizializzazione di OpenGL
        glEnable(GL_TEXTURE_2D);
        glGenTextures(1, &textureID);
        glBindTexture(GL_TEXTURE_2D, textureID);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    simulazione(
    int world_dim, int n_creature, 
    int *dims_model, int n_layer, size_t reserve_free_memory, 
    float *weights_models, float *biases_models, 
    int const N_EPHOCS, int const N_STEPS, int const MAX_WORKSPACE, int const METHOD_EVAL, GL_TEXTURE_2D)
    clock_t end = clock();  // End time
    std::cout << "Tempo esecuzione programma: " << (end - start) / CLOCKS_PER_SEC << " secondi" << std::endl;


    
    return 0;
}