CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2

all: projet

projet: main.o network.o metrics.o
	$(CXX) $(CXXFLAGS) -o projet main.o network.o metrics.o

main.o: main.cpp network.h metrics.h
	$(CXX) $(CXXFLAGS) -c main.cpp

network.o: network.cpp network.h
	$(CXX) $(CXXFLAGS) -c network.cpp

metrics.o: metrics.cpp metrics.h network.h
	$(CXX) $(CXXFLAGS) -c metrics.cpp

clean:
	rm -f *.o projet
