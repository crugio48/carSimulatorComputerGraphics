CFLAGS = -std=c++17 -O2 -I$(HEADERS_INCLUDE_PATH)

LDFLAGS = -lglfw -lvulkan -ldl -lpthread -lX11 -lXrandr -lXi -lXxf86vm

HEADERS_INCLUDE_PATH = ./headers

carSim_exec: carSimulator.cpp
	glslc shaders/shader.vert -o shaders/vert.spv
	glslc shaders/carDay.frag -o shaders/car_day_frag.spv
	glslc shaders/terrainDay.frag -o shaders/terrain_day_frag.spv
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
