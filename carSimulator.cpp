#include "VulkanApp.hpp"

const std::string CAR_MODEL_PATH = "models/car m1.obj";
const std::string CAR_TEXTURE_PATH = "textures/car m1 texture.png";

const std::string TERRAIN_MODEL_PATH = "models/flatTerrain.obj";
const std::string TERRAIN_TEXTURE_PATH = "textures/terrainTex.jpeg";

const std::string VERTEX_SHADER_PATH = "shaders/vert.spv";
const std::string FRAGMENT_SHADER_PATH = "shaders/frag.spv";


// The uniform buffer objects used
struct GlobalUniformBufferObject {
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct UniformBufferObject {
	alignas(16) glm::mat4 model;
};


//movement struct
struct MovementInfo {
	glm::vec3 cameraPosition;
	glm::vec3 carPosition;
	glm::vec3 upVector;

	float acceleration;
	float velocity;

	glm::mat4 carRotationMatrix;
	glm::vec3 carDirection;

	void init(){
		carPosition = glm::vec3(0,0,0);
		cameraPosition = carPosition + glm::vec3(0, 2, 2);
		upVector = glm::vec3(0,1,0);

		acceleration = 0;
		velocity = 0;

		carRotationMatrix = glm::rotate(glm::mat4(1.0), glm::radians(-90.0f), glm::vec3(1,0,0)); //initial facing direction
		carDirection = glm::vec3(0,0,-1);
	}
};


class MyProject : public BaseProject {
	protected:
	// Here you list all the Vulkan objects you need:
	
	// Descriptor Layouts [what will be passed to the shaders]
	DescriptorSetLayout globalDSL;
	DescriptorSetLayout objectDSL;

	// Pipelines [Shader couples]
	Pipeline pipeline;

	// Models, textures and Descriptors (values assigned to the uniforms)
	DescriptorSet globalDescriptorSet;

	Model carModel;
	Texture carTexture;
	DescriptorSet carDescriptorSet;

	Model terrainModel;
	Texture terrainTexture;
	DescriptorSet terrainDescriptorSet;
	

	//extra
	MovementInfo movInfo;

    /**
     * @brief Set the Window Parameters and the pool sizes
     * 
     */
	void setWindowParameters() {
		// window size, title and initial background
		windowWidth = 1600;
		windowHeight = 1200;
		windowTitle = "Car simulator";
		initialBackgroundColor = {0.0f, 0.0f, 0.0f, 1.0f};
		
		// Descriptor pool sizes
		uniformBlocksInPool = 3;        //how many descriptor sets??
		texturesInPool = 2;
		setsInPool = 3;
	}
	
	/**
	 * @brief initialize the struct elements of the application
	 * 
	 */
	void localInit() {
		// Descriptor Layouts [what will be passed to the shaders]
		globalDSL.init(this, {
								// this array contains the binding:
								// first  element : the binding number
								// second element : the time of element (buffer or texture)
								// third  element : the pipeline stage where it will be used
								{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_ALL_GRAPHICS}	
							 });


		objectDSL.init(this, {
								{0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT},
								{1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_SHADER_STAGE_FRAGMENT_BIT}
							 });

		// Pipelines [Shader couples]
		// The last array, is a vector of pointer to the layouts of the sets that will
		// be used in this pipeline. The first element will be set 0, and so on..
		pipeline.init(this, VERTEX_SHADER_PATH, FRAGMENT_SHADER_PATH, {&globalDSL, &objectDSL});

		// Models, textures and Descriptors (values assigned to the uniforms)
		carModel.init(this, CAR_MODEL_PATH);
		carTexture.init(this, CAR_TEXTURE_PATH);
		carDescriptorSet.init(this, &objectDSL, {
		// the second parameter, is a pointer to the Uniform Set Layout of this set
		// the last parameter is an array, with one element per binding of the set.
		// first  elmenet : the binding number
		// second element : UNIFORM or TEXTURE (an enum) depending on the type
		// third  element : only for UNIFORMs, the size of the corresponding C++ object
		// fourth element : only for TEXTUREs, the pointer to the corresponding texture object
					{0, UNIFORM, sizeof(UniformBufferObject), nullptr},
					{1, TEXTURE, 0, &carTexture}
				});

		terrainModel.init(this, TERRAIN_MODEL_PATH);
		terrainTexture.init(this, TERRAIN_TEXTURE_PATH);
		terrainDescriptorSet.init(this, &objectDSL, {
					{0, UNIFORM, sizeof(UniformBufferObject), nullptr},
					{1, TEXTURE, 0, &terrainTexture}
				});

		
		globalDescriptorSet.init(this, &globalDSL, {
			{0, UNIFORM, sizeof(GlobalUniformBufferObject), nullptr}
		});


		//extra
		movInfo.init();
	}

	/**
	 * @brief release the resources aquired
	 * 
	 */
	void localCleanup() {
		carDescriptorSet.cleanup();
		carTexture.cleanup();
		carModel.cleanup();

		terrainDescriptorSet.cleanup();
		terrainTexture.cleanup();
		terrainModel.cleanup();

		globalDescriptorSet.cleanup();

		pipeline.cleanup();
		
		globalDSL.cleanup();
		objectDSL.cleanup();
	}
	
	/**
	 * @brief declare all the commands needed to draw all elements (command buffer initialized only at the start)
	 * 
	 * @param commandBuffer 
	 * @param currentImage 
	 */
	void populateCommandBuffer(VkCommandBuffer commandBuffer, int currentImage) {
				
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				pipeline.graphicsPipeline);

		///////////////////////////////                    GLOBAL

		vkCmdBindDescriptorSets(commandBuffer,
						VK_PIPELINE_BIND_POINT_GRAPHICS,
						pipeline.pipelineLayout, 0, 1, &globalDescriptorSet.descriptorSets[currentImage],
						0, nullptr);

				
		//////////////////////////////                            CAR 
		VkBuffer vertexBuffersCar[] = {carModel.vertexBuffer};
		// property .vertexBuffer of models, contains the VkBuffer handle to its vertex buffer
		VkDeviceSize offsetsCar[] = {0};
		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffersCar, offsetsCar);
		// property .indexBuffer of models, contains the VkBuffer handle to its index buffer
		vkCmdBindIndexBuffer(commandBuffer, carModel.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		// property .pipelineLayout of a pipeline contains its layout.
		// property .descriptorSets of a descriptor set contains its elements.
		vkCmdBindDescriptorSets(commandBuffer,
						        VK_PIPELINE_BIND_POINT_GRAPHICS,
						        pipeline.pipelineLayout, 1, 1, &carDescriptorSet.descriptorSets[currentImage],
						        0, nullptr);

		// property .indices.size() of models, contains the number of triangles * 3 of the mesh.
		vkCmdDrawIndexed(commandBuffer,
					     static_cast<uint32_t>(carModel.indices.size()), 1, 0, 0, 0);

		///////////////////////////////                   TERRAIN

		VkBuffer vertexBuffersTerrain[] = {terrainModel.vertexBuffer};
		VkDeviceSize offsetsTerrain[] = {0};

		vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffersTerrain, offsetsTerrain);
		vkCmdBindIndexBuffer(commandBuffer, terrainModel.indexBuffer, 0, VK_INDEX_TYPE_UINT32);

		vkCmdBindDescriptorSets(commandBuffer,
								VK_PIPELINE_BIND_POINT_GRAPHICS,
						        pipeline.pipelineLayout, 1, 1, &terrainDescriptorSet.descriptorSets[currentImage],
						        0, nullptr);

		vkCmdDrawIndexed(commandBuffer,
						 static_cast<uint32_t>(terrainModel.indices.size()), 1, 0, 0, 0);
						
	}

	/**
	 * @brief update uniform buffers to move and animate
	 * 
	 * @param currentImage 
	 */
	void updateUniformBuffer(uint32_t currentImage) {
		static auto startTime = std::chrono::high_resolution_clock::now();
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>
					(currentTime - startTime).count();

		static float aspectRatio = ((float) swapChainExtent.width) / (float) swapChainExtent.height;


		
		updateMovementInfo(time);

					
		GlobalUniformBufferObject gubo{};
		UniformBufferObject ubo{};
		void* data;

		gubo.view = glm::lookAt(movInfo.cameraPosition, movInfo.carPosition, movInfo.upVector);

		gubo.proj = glm::perspective(glm::radians(90.0f), 
									aspectRatio, 
									0.1f, 
									10.0f);
		gubo.proj[1][1] *= -1;

		// Here is where you actually update your uniforms
		vkMapMemory(device, globalDescriptorSet.uniformBuffersMemory[0][currentImage], 0, sizeof(gubo), 0, &data);
			memcpy(data, &gubo, sizeof(gubo));
		vkUnmapMemory(device, globalDescriptorSet.uniformBuffersMemory[0][currentImage]);

		
		///////////////////////////////////   car movement


		ubo.model = glm::translate(glm::mat4(1), movInfo.carPosition) * 
					movInfo.carRotationMatrix *
					glm::scale(glm::mat4(1), glm::vec3(0.4,0.4,0.4));

		vkMapMemory(device, carDescriptorSet.uniformBuffersMemory[0][currentImage], 0, sizeof(ubo), 0, &data);
		    memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, carDescriptorSet.uniformBuffersMemory[0][currentImage]);



		///////////////////////////// terrain 

		ubo.model = glm::translate(glm::mat4(1), glm::vec3(0,-3,0)) * 
					glm::rotate(glm::mat4(1), glm::radians(-90.0f), glm::vec3(1,0,0)) *
					glm::scale(glm::mat4(1), glm::vec3(10,10,10));

		vkMapMemory(device, terrainDescriptorSet.uniformBuffersMemory[0][currentImage], 0, sizeof(ubo), 0, &data);
		    memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, terrainDescriptorSet.uniformBuffersMemory[0][currentImage]);


	}	


	/**
	 * @brief function to update the movement info at each frame
	 * 
	 */
	void updateMovementInfo(float deltaTime) {

		glm::vec3 oldCarPosition = movInfo.carPosition;
		glm::vec3 oldCarDirection = movInfo.carDirection;

		//user input
		int A = glfwGetKey(window, GLFW_KEY_A);  // true(1) if button A pressed
		int D = glfwGetKey(window, GLFW_KEY_D);  // true(1) if button D pressed
		int W = glfwGetKey(window, GLFW_KEY_W);  // true(1) if button W pressed
		int S = glfwGetKey(window, GLFW_KEY_S);  // true(1) if button S pressed


		//compute acceleration
		if (W && !S) {
			movInfo.acceleration = +10;
		}
		else if (S && !W) {
			movInfo.acceleration = -10;
		}
		else {
			movInfo.acceleration = 0;
		}


		//compute velocity
		movInfo.velocity += movInfo.acceleration * deltaTime;

		//simulating friction
		if (movInfo.velocity >= 0.3 || movInfo.velocity <= -0.3) {
			movInfo.velocity *= 0.99;
		}
		else {
			movInfo.velocity = 0;
		}

		//compute position and direction
		if (movInfo.velocity != 0) {

		}



		
		int upArrow = glfwGetKey(window, GLFW_KEY_UP);  // true(1) if button up arrow pressed
		int downArrow = glfwGetKey(window, GLFW_KEY_DOWN);  // true(1) if button down arrow pressed
		int leftArrow = glfwGetKey(window, GLFW_KEY_LEFT);  // true(1) if button left arrow pressed
		int rightArrow = glfwGetKey(window, GLFW_KEY_RIGHT);  // true(1) if button right arrow pressed

		if (upArrow) {
			movInfo.cameraPosition = movInfo.carPosition + glm::vec3(0,2,-2);
		}
		else if (downArrow) {
			movInfo.cameraPosition = movInfo.carPosition + glm::vec3(0,2,2);
		}
		else if (leftArrow) {
			movInfo.cameraPosition = movInfo.carPosition + glm::vec3(-2,2,0);
		}
		else if (rightArrow) {
			movInfo.cameraPosition = movInfo.carPosition + glm::vec3(2,2,0);
		}
		else {
			movInfo.cameraPosition = movInfo.carPosition + glm::vec3(0,2,2);
		}

	}
};

// This is the main: probably you do not need to touch this!
int main() {
    MyProject app;

    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}