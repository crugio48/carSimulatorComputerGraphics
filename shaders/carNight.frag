#version 450

layout(set=1, binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragViewDir;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragPos;

layout(location = 0) out vec4 outColor;

/* 
	Lambert diffuse BRDF model

	parameters:
		vec3 L : light direction
		vec3 N : normal vector
		vec3 V : view direction
		vec3 C : main color (diffuse color)
*/
vec3 Lambert_Diffuse_BRDF(vec3 L, vec3 N, vec3 V, vec3 C) {

	vec3 returnColor = C * max( dot( L , N ) , 0 );
	
	return returnColor;
}

/*
	Phong Specular BRDF model

	parameters:
		vec3 L : light direction
		vec3 N : normal vector
		vec3 V : view direction
		vec3 C : main color (specular color)
		float gamma : exponent of the cosine term
*/
vec3 Phong_Specular_BRDF(vec3 L, vec3 N, vec3 V, vec3 C, float gamma)  {

	vec3 reflectedVector = - reflect(L, N);

	vec3 returnColor = C * pow( clamp( dot(V , reflectedVector), 0, 1 ), gamma );
	
	return returnColor;
}

void main() {
	//make sure direct light dir and color and ambient light is the same for both day fragment shaders
    vec3 directLightDirection = normalize(vec3(0.2, 1.0, 0.2)); 
	vec3 directLightColor = vec3(0.1, 0.1, 0.1);
	vec3 ambientLight = vec3(0.01,0.01, 0.01);


	vec3  diffColor = texture(texSampler, fragTexCoord).rgb;
	vec3  specColor = vec3(0.01, 0.01, 0.01);
	float specPower = 200.0;
	
	vec3 normal = normalize(fragNorm);
	vec3 viewDir = normalize(fragViewDir);

	
	// Lambert diffuse
	vec3 diffuse  = Lambert_Diffuse_BRDF(directLightDirection, normal, viewDir, diffColor);
	// Phong specular
	vec3 specular = Phong_Specular_BRDF(directLightDirection, normal, viewDir, specColor, specPower);
	// ambient lighting
	vec3 ambient  = ambientLight * diffColor;


	vec3 color = directLightColor * (diffuse + specular);

	//adding ambient and clamping
	color = clamp((color + ambient), vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
	
	outColor = vec4(color, 1.0);
}