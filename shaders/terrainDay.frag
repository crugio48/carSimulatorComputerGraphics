#version 450

layout(set=1, binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragViewDir;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;


/*
    Oren nayar diffuse model

    parameters:
        vec3 L : light direction
		vec3 N : normal vector
		vec3 V : view direction
		vec3 C : main color (diffuse color)
        float sigma : roughness of the material. range: (0 , pi/2)

*/
vec3 Oren_Nayar_Diffuse_BRDF(vec3 L, vec3 N, vec3 V, vec3 C, float sigma) {

	float theta1, theta2, alpha, beta;
	theta1 = acos( dot(L, N) );
	theta2 = acos( dot(V, N) );
	alpha = max(theta1, theta2);
	beta = min(theta1, theta2);

	float a, b;
	a = 1.0 - 0.5 * ( pow(sigma, 2) / ( pow(sigma, 2) + 0.33 ) );
	b = 0.45 * ( pow(sigma, 2) / ( pow(sigma, 2) + 0.09 ) );

	vec3 v1, v2;
	v1 = normalize( L - dot(L, N) * N );
	v2 = normalize( V - dot(V, N) * N );

	float g = max(0, dot(v1, v2));

	vec3 l = C * clamp( dot(L, N), 0, 1);

	vec3 returnColor = l * ( a + b * g * sin(alpha) * tan(beta));

	return returnColor;
}


void main() {
    //make sure direct light dir and color and ambient light is the same for both day fragment shaders
    vec3 directLightDirection = normalize(vec3(0.2, 1.0, 0.2)); 
	vec3 directLightColor = vec3(1.0, 1.0, 1.0);
    vec3 ambientLight = vec3(0.3,0.3, 0.3);


	vec3  diffColor = texture(texSampler, fragTexCoord).rgb;
	float roughness = 1.0;
	
	vec3 normal = normalize(fragNorm);
	vec3 viewDir = normalize(fragViewDir);

	
	// Lambert diffuse
	vec3 diffuse  = Oren_Nayar_Diffuse_BRDF(directLightDirection, normal, viewDir, diffColor, roughness);
	// ambient lighting
	vec3 ambient  = ambientLight * diffColor;


	vec3 color = directLightColor * (diffuse);

	//adding ambient and clamping
	color = clamp((color + ambient), vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
	
	outColor = vec4(color, 1.0);
}