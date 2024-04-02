# -*- MakeFile -*-

SRC=./src/
INC=./include/
BIN=./bin/
OBJ=./obj/

build: $(OBJ)main.o $(OBJ)dense.o $(OBJ)activation.o $(OBJ)net.o \
	$(OBJ)cost.o $(OBJ)gradient.o $(OBJ)optimizer.o
	g++ $(OBJ)main.o $(OBJ)dense.o $(OBJ)activation.o $(OBJ)net.o $(OBJ)cost.o \
		$(OBJ)gradient.o $(OBJ)optimizer.o -o $(BIN)main -O2 -larmadillo

$(OBJ)main.o: $(SRC)main.cpp
	g++ -c -I$(INC) $(SRC)main.cpp -o $(OBJ)main.o

$(OBJ)dense.o: $(SRC)dense.cpp $(INC)dense.h
	g++ -c -I$(INC) $(SRC)dense.cpp -o $(OBJ)dense.o

$(OBJ)activation.o: $(SRC)activation.cpp $(INC)activation.h
	g++ -c -I$(INC) $(SRC)activation.cpp -o $(OBJ)activation.o

$(OBJ)net.o: $(SRC)net.cpp $(INC)net.h
	g++ -c -I$(INC) $(SRC)net.cpp -o $(OBJ)net.o

$(OBJ)cost.o: $(SRC)cost.cpp $(INC)cost.h
	g++ -c -I$(INC) $(SRC)cost.cpp -o $(OBJ)cost.o

$(OBJ)gradient.o: $(SRC)gradient.cpp $(INC)gradient.h
	g++ -c -I$(INC) $(SRC)gradient.cpp -o $(OBJ)gradient.o

$(OBJ)optimizer.o: $(SRC)optimizer.cpp $(INC)optimizer.h
	g++ -c -I$(INC) $(SRC)optimizer.cpp -o $(OBJ)optimizer.o

clean:
	rm $(BIN)* $(OBJ)*
