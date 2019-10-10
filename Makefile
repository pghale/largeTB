all: withpetsc2.cpp
	CC -g -o tx withpetsc2.cpp

test: withpetsc2.cpp
	rm t1
	CC -g -o t1 withpetsc2.cpp
	aprun -n 1 ./t1 4 4 4
	aprun -n 10 ./t1 10 10 10

t2: withSP2.cpp
	CC -g -o t2 withSP2.cpp 

scf: withSP2_deltaH.cpp
	CC  -hstd=c++11  -g -o scf withSP2_deltaH.cpp

clean: 
	rm scf
	rm t2
