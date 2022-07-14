#include <vulkan/vulkan.hpp>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

class CarSimulatorApp {
    public:
        void run() {
            initVulkan();
            mainLoop();
            cleanup();
        }
    
    private:
        void initVulkan() {

        }

        void mainLoop() {

        }

        void cleanup() {

        }
};

int main() {
    CarSimulatorApp app;

    try {
        app.run();

    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    return EXIT_SUCCESS;
}