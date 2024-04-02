# -*- MakeFile -*-

SRC=./src/
INC=./include/
BIN=./bin/
OBJ=./obj/

build: $(OBJ)main.o $(OBJ)tinynet.o
	gcc $(OBJ)main.o $(OBJ)tinynet.o  -o $(BIN)main -lm

$(OBJ)main.o: $(SRC)main.c
	gcc -c -I$(INC) $(SRC)main.c -o $(OBJ)main.o

$(OBJ)tinynet.o: $(SRC)tinynet.c $(INC)tinynet.h
	gcc -c -I$(INC) $(SRC)tinynet.c -o $(OBJ)tinynet.o

clean:
	rm $(BIN)* $(OBJ)*
