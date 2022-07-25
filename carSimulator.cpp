#include "VulkanApp.hpp"

const std::string CAR_MODEL_PATH = "models/car m1 v3.obj";
const std::string CAR_TEXTURE_PATH = "textures/car m1 texture.png";

const std::string TERRAIN_MODEL_PATH = "models/final 3d map.obj";
const std::string TERRAIN_TEXTURE_PATH = "textures/terrain texture.png";

const std::string VERTEX_SHADER_PATH = "shaders/vert.spv";
const std::string FRAGMENT_SHADER_PATH = "shaders/frag.spv";

const std::string HEIGHT_MAP_PATH = "maps/height map.png";
const std::string NORMAL_MAP_PATH = "maps/normal map.png";

const int MAP_SCALE = 1000;
const float MAP_HEIGHT = 0.35;
const float CAR_SCALE = 0.4;

const float CAMERA_DISTANCE = 3.0;
const float CAMERA_HEIGHT = 1.0;


// The uniform buffer objects used
struct GlobalUniformBufferObject {
	alignas(16) glm::mat4 view;
	alignas(16) glm::mat4 proj;
};

struct UniformBufferObject {
	alignas(16) glm::mat4 model;
};


//Struct for height and normal map
struct ImageInfo {

	stbi_uc* pixels;
	int imageWidth, imageHeight, imageChannels;

    float mapHeight;
	float mapWidthX, mapWidthZ;  //half width

	void initImage(std::string file, float maxHeightOfMap, float widthOfMapX, float widthOfMapZ) {
		pixels = stbi_load(file.c_str(), &imageWidth, &imageHeight, &imageChannels, STBI_rgb_alpha);

		if (!pixels) {
			throw std::runtime_error("failed to load texture image!");
		}

        mapHeight = maxHeightOfMap;
		mapWidthX = widthOfMapX;
		mapWidthZ = widthOfMapZ;
	}
	
	void getPixelColors(size_t x, size_t y, float *red, float *green, float *blue) {
		stbi_uc r = pixels[4 * (y * imageWidth + x) + 0];
		stbi_uc g = pixels[4 * (y * imageWidth + x) + 1];
		stbi_uc b = pixels[4 * (y * imageWidth + x) + 2];

        *red = ((float) r) / 255.0f;
        *green = ((float) g) / 255.0f;
        *blue = ((float) b) / 255.0f;

	}

	glm::vec3 getNormalVector(float xCoord, float zCoord) {

		float temp = xCoord + mapWidthX;
		size_t x = std::ceil(temp * ((float) imageWidth / (mapWidthX * 2)));

		temp = zCoord + mapWidthZ;
		size_t y = std::ceil(temp * ((float) imageHeight / (mapWidthZ * 2)));

		stbi_uc r = pixels[4 * (y * imageWidth + x) + 0];
		stbi_uc g = pixels[4 * (y * imageWidth + x) + 1];
		stbi_uc b = pixels[4 * (y * imageWidth + x) + 2];


		glm::vec3 returnVector = glm::vec3( 
			((float) r / 255.0f) * 2 - 1.0f,
			((float) g / 255.0f) * 2 - 1.0f,
			((float) b / 255.0f) * 2 - 1.0f
		);

		returnVector = glm::normalize(returnVector);

		return returnVector;
	}

	float getHeightValue(float xCoord, float zCoord) {

		float temp = xCoord + mapWidthX;
		size_t x = std::ceil(temp * ((float) imageWidth / (mapWidthX * 2)));

		temp = zCoord + mapWidthZ;
		size_t y = std::ceil(temp * ((float) imageHeight / (mapWidthZ * 2)));


		stbi_uc h = pixels[4 * (y * imageWidth + x) + 0];
		//*g = pixels[4 * (y * imageWidth + x) + 1];
		//*b = pixels[4 * (y * imageWidth + x) + 2];

		//fo height map r,g,b should all have the same value

		float height = ((float) h) * (mapHeight / 255.0f);

        return height;
	}

	void cleanup(){
		stbi_image_free(pixels);
	}
};

//movement struct
struct MovementInfo {
	glm::vec3 cameraPosition;
	glm::vec3 carPosition;
	glm::vec3 upVector;

	float acceleration;
	float velocity;

	float angX;    //pitch
	float angY;    //yaw
	float angZ;    //roll
	glm::mat4 carRotation;
	glm::vec3 carDirection;


	glm::mat4 terrainTransform; //to keep const value of terrain placement

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

	ImageInfo heightMap;
	ImageInfo normalMap;
	
    /**
     * @brief Set the Window Parameters and the pool sizes
     * 
     */
	void setWindowParameters() {
		// window size, title and initial background
		windowWidth = 1000;
		windowHeight = 1000;
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
		heightMap.initImage(HEIGHT_MAP_PATH, MAP_HEIGHT * (float) MAP_SCALE, MAP_SCALE, MAP_SCALE);
		normalMap.initImage(NORMAL_MAP_PATH, MAP_HEIGHT * (float) MAP_SCALE, MAP_SCALE, MAP_SCALE);

		
		setInitialPosition();
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

		heightMap.cleanup();
		normalMap.cleanup();
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
		static float lastTime = 0;
		auto currentTime = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>
					(currentTime - startTime).count();
		float deltaT = time - lastTime;
		lastTime = time;

		static float aspectRatio = ((float) swapChainExtent.width) / (float) swapChainExtent.height;
		static float nearPlane = 1;
		static float farPlane = 100;


		
		updateMovementInfo(deltaT);

					
		GlobalUniformBufferObject gubo{};
		UniformBufferObject ubo{};
		void* data;

		gubo.view = glm::lookAt(movInfo.cameraPosition, movInfo.carPosition, movInfo.upVector);

		gubo.proj = glm::perspective(glm::radians(90.0f), 
									aspectRatio,
									nearPlane,
									farPlane);
		gubo.proj[1][1] *= -1;

		// Here is where you actually update your uniforms
		vkMapMemory(device, globalDescriptorSet.uniformBuffersMemory[0][currentImage], 0, sizeof(gubo), 0, &data);
			memcpy(data, &gubo, sizeof(gubo));
		vkUnmapMemory(device, globalDescriptorSet.uniformBuffersMemory[0][currentImage]);

		
		///////////////////////////////////   car movement


		ubo.model = glm::translate(glm::mat4(1), movInfo.carPosition) * 
					movInfo.carRotation *
					glm::scale(glm::mat4(1), glm::vec3(CAR_SCALE, CAR_SCALE, CAR_SCALE));

		vkMapMemory(device, carDescriptorSet.uniformBuffersMemory[0][currentImage], 0, sizeof(ubo), 0, &data);
		    memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, carDescriptorSet.uniformBuffersMemory[0][currentImage]);



		///////////////////////////// terrain 

		ubo.model = movInfo.terrainTransform;

		vkMapMemory(device, terrainDescriptorSet.uniformBuffersMemory[0][currentImage], 0, sizeof(ubo), 0, &data);
		    memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(device, terrainDescriptorSet.uniformBuffersMemory[0][currentImage]);


	}	


	/**
	 * @brief Set the Initial Position of car and initialize all movement info parameters
	 * 
	 */
	void setInitialPosition() {

		movInfo.carPosition = glm::vec3(
			-980,
			heightMap.getHeightValue(-980,980),
			980
		);

		movInfo.carDirection = glm::vec3(0,0,-1);   //initial direction

		movInfo.cameraPosition = movInfo.carPosition + -CAMERA_DISTANCE * movInfo.carDirection + glm::vec3(0,CAMERA_HEIGHT,0);
		
		movInfo.upVector = glm::vec3(0,1,0);

		movInfo.acceleration = 0;
		movInfo.velocity = 0;

		glm::vec3 terrainNormal = normalMap.getNormalVector(movInfo.carPosition.x,movInfo.carPosition.z);
		movInfo.angX = glm::acos(glm::dot(terrainNormal, glm::normalize(
			glm::vec3(
				terrainNormal.x,
				terrainNormal.y,
				0
				))));
		movInfo.angY = 0;
		movInfo.angZ = glm::acos(glm::dot(terrainNormal, glm::normalize(
			glm::vec3(
				0,
				terrainNormal.y,
				terrainNormal.z
				))));
		movInfo.carRotation= glm::rotate(glm::mat4(1), movInfo.angZ, glm::vec3(0,0,1)) *
					 glm::rotate(glm::mat4(1), movInfo.angX, glm::vec3(1,0,0)) *
					 glm::rotate(glm::mat4(1), glm::radians(movInfo.angY), glm::vec3(0,1,0)); //initial rotation

		

		movInfo.terrainTransform = glm::translate(glm::mat4(1), glm::vec3(0,0,0)) * 
						   glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(1,0,0)) *
						   glm::scale(glm::mat4(1), glm::vec3(MAP_SCALE, MAP_SCALE, MAP_SCALE));
	}


	/**
	 * @brief function to update the movement info at each frame
	 * 
	 */
	void updateMovementInfo(float deltaTime) {

		float rotStep = deltaTime * 45;

		//user input
		int A = glfwGetKey(window, GLFW_KEY_A);  // true(1) if button A pressed
		int D = glfwGetKey(window, GLFW_KEY_D);  // true(1) if button D pressed
		int W = glfwGetKey(window, GLFW_KEY_W);  // true(1) if button W pressed
		int S = glfwGetKey(window, GLFW_KEY_S);  // true(1) if button S pressed


		//compute acceleration
		if (W and !S) {
			movInfo.acceleration = +10;
		}
		else if (S and !W) {
			movInfo.acceleration = -10;
		}
		else {
			movInfo.acceleration = 0;
		}

		//compute velocity
		movInfo.velocity += movInfo.acceleration * deltaTime;


		//simulating friction
		if (movInfo.acceleration == 0) {

			if (movInfo.velocity >= 0.5 or movInfo.velocity <= -0.5) {
				movInfo.velocity *= 0.9999;
			}
			else {
				movInfo.velocity = 0;
			}
		}


		//compute direction
		if (movInfo.velocity > 0) {
			if (A and !D) {
				movInfo.angY += rotStep;
			}
			else if (D and !A) {
				movInfo.angY -= rotStep;
			}
		}
		else if (movInfo.velocity < 0) {
			if (A and !D) {
				movInfo.angY -= rotStep;
			}
			else if (D and !A) {
				movInfo.angY += rotStep;
			}
		}

		//compute direction
		glm::vec4 tempDirection = glm::vec4(0,0,-1,0);   //starting position of direction
		tempDirection = glm::rotate(glm::mat4(1), glm::radians(movInfo.angZ), glm::vec3(0,0,1))*
						glm::rotate(glm::mat4(1), glm::radians(movInfo.angX), glm::vec3(1,0,0))*
					    glm::rotate(glm::mat4(1), glm::radians(movInfo.angY), glm::vec3(0,1,0))*
					    tempDirection;
		
		movInfo.carDirection.x = tempDirection.x;
		movInfo.carDirection.y = tempDirection.y;
		movInfo.carDirection.z = tempDirection.z;

		movInfo.carDirection = glm::normalize(movInfo.carDirection);



		//compute position
		movInfo.carPosition += movInfo.velocity * deltaTime * movInfo.carDirection;

		movInfo.carPosition = glm::vec3(
			movInfo.carPosition.x,
			heightMap.getHeightValue(movInfo.carPosition.x, movInfo.carPosition.z),
			movInfo.carPosition.z
		);



		//compute angX and angZ with terrainNormal
		glm::vec3 terrainNormal = normalMap.getNormalVector(movInfo.carPosition.x,movInfo.carPosition.z);
		movInfo.angX = glm::acos(glm::dot(terrainNormal, glm::normalize(
			glm::vec3(
				terrainNormal.x,
				terrainNormal.y,
				0
				))));
		movInfo.angZ = glm::acos(glm::dot(terrainNormal, glm::normalize(
			glm::vec3(
				0,
				terrainNormal.y,
				terrainNormal.z
				))));



		//update rotation
		movInfo.carRotation = glm::rotate(glm::mat4(1), movInfo.angZ, glm::vec3(0,0,1))*
							  glm::rotate(glm::mat4(1), movInfo.angX, glm::vec3(1,0,0))*
							  glm::rotate(glm::mat4(1), glm::radians(movInfo.angY), glm::vec3(0,1,0));
		
		



		//get camera controller input
		int upArrow = glfwGetKey(window, GLFW_KEY_UP);  // true(1) if button up arrow pressed
		int downArrow = glfwGetKey(window, GLFW_KEY_DOWN);  // true(1) if button down arrow pressed
		int leftArrow = glfwGetKey(window, GLFW_KEY_LEFT);  // true(1) if button left arrow pressed
		int rightArrow = glfwGetKey(window, GLFW_KEY_RIGHT);  // true(1) if button right arrow pressed

		//update camera position
		if (upArrow) {
			movInfo.cameraPosition = movInfo.carPosition + CAMERA_DISTANCE * movInfo.carDirection + glm::vec3(0,CAMERA_HEIGHT,0);
		}
		else if (downArrow) {
			movInfo.cameraPosition = movInfo.carPosition + -CAMERA_DISTANCE * movInfo.carDirection + glm::vec3(0,CAMERA_HEIGHT,0);
		}
		else if (leftArrow) {
			tempDirection = glm::vec4(movInfo.carDirection.x,
									  movInfo.carDirection.y,
									  movInfo.carDirection.z,
									  0);
			tempDirection = glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0,1,0))*
							tempDirection;
			glm::vec3 perpendicularDirection = glm::vec3(tempDirection.x,
														 tempDirection.y,
														 tempDirection.z);
			perpendicularDirection = glm::normalize(perpendicularDirection);

			movInfo.cameraPosition = movInfo.carPosition + CAMERA_DISTANCE * perpendicularDirection + glm::vec3(0,CAMERA_HEIGHT,0);
		}
		else if (rightArrow) {
			tempDirection = glm::vec4(movInfo.carDirection.x,
									  movInfo.carDirection.y,
									  movInfo.carDirection.z,
									  0);
			tempDirection = glm::rotate(glm::mat4(1), glm::radians(90.0f), glm::vec3(0,1,0))*
							tempDirection;
			glm::vec3 perpendicularDirection = glm::vec3(tempDirection.x,
														 tempDirection.y,
														 tempDirection.z);
			perpendicularDirection = glm::normalize(perpendicularDirection);

			movInfo.cameraPosition = movInfo.carPosition + -CAMERA_DISTANCE * perpendicularDirection + glm::vec3(0,CAMERA_HEIGHT,0);
		}
		else {
			movInfo.cameraPosition = movInfo.carPosition + -CAMERA_DISTANCE * movInfo.carDirection + glm::vec3(0,CAMERA_HEIGHT,0);
		}


		//std::cout << movInfo.velocity << "\n";
		//std::cout << movInfo.carDirection.x << " " << movInfo.carDirection.y << " " << movInfo.carDirection.z << " " << "\n";
		//std::cout << movInfo.carPosition.x << " " << movInfo.carPosition.y << " " << movInfo.carPosition.z << " " << "\n";

		//std::cout << terrainNormal.x << " " << terrainNormal.y << " " << terrainNormal.z << " " << "\n";
		//std::cout << movInfo.angX << " " << movInfo.angY << " " << movInfo.angZ << " " << "\n";
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