CFLAGS = -std=c++17 -O2

LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi -lXxf86vm

carSim_exec: carSimulator.cpp
	g++ $(CFLAGS) -o carSim_exec carSimulator.cpp $(LDFLAGS)
	
	
.PHONY: test clean restart

test: carSim_exec
	./carSim_exec

clean:
	rm -f carSim_exec

restart:
	make --no-print-directory clean
	make --no-print-directory
	make --no-print-directory test
