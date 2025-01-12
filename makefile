#nvcc -arch=sm_50 main.cpp libs/kernel.cu -o main
#./main

# Variabili per i percorsi e i file
CC = nvcc
CFLAGS = -arch=sm_50

SRC_CPP = main.cpp
SRC_CU = libs/kernel.cu libs/loader.cpp
OBJ_CPP = main.o
OBJ_CU = kernel.o loader.o
OUT = main

# Target di compilazione
all: $(OUT)

# Regola per compilare il file main.cpp
$(OBJ_CPP): $(SRC_CPP)
	$(CC) $(CFLAGS) -c $(SRC_CPP) -o $(OBJ_CPP)

# Regola per compilare il file kernel.cu
$(OBJ_CU): $(SRC_CU)
	$(CC) $(CFLAGS) -c $(SRC_CU) -o $(OBJ_CU)

# Linka i file oggetto e crea l'eseguibile
$(OUT): $(OBJ_CPP) $(OBJ_CU)
	$(CC) $(OBJ_CPP) $(OBJ_CU) -o $(OUT)

# Pulizia dei file oggetto e dell'eseguibile
clean:
	rm -f $(OBJ_CPP) $(OBJ_CU) $(OUT)

# Regola per eseguire il programma
run: $(OUT)
	./$(OUT)
