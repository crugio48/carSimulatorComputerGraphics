CFLAGS = -std=c++17 -O2 -I$(HEADERS_INCLUDE_PATH)

LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi -lXxf86vm

HEADERS_INCLUDE_PATH = ./headers

carSim_exec: carSimulator.cpp
	glslc shaders/car.vert -o shaders/car_vert.spv
	glslc shaders/car.frag -o shaders/car_frag.spv
	glslc shaders/terrain.vert -o shaders/terrain_vert.spv
	glslc shaders/terrain.frag -o shaders/terrain_frag.spv
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
