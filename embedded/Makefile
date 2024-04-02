#-*- Makefile -*-
# Comment tells editor to format like Makefile

# Format of file:
# target: dependencies
#	action

# I think this is called a dependency tree
all: test

test.o: test.c utils.h
	gcc -c test.c

utils.o: utils.c
	gcc -c utils.c

test: test.o utils.o
	gcc test.o utils.o -o feedForward -lgsl -lgslcblas -lm

clean:
	rm *.o feedForward
