#version 450

layout(set = 0, binding = 0) uniform globalUniformBufferObject {
	mat4 view;
	mat4 proj;
	vec3 rightFrontLightPos;
	vec3 leftFrontLightPos;
	vec3 carLightDir;
    vec3 rightRearLightPos;
	vec3 leftRearLightPos;
    vec3 backLightsColor;
} gubo;

layout(set=1, binding = 1) uniform sampler2D texSampler;

layout(location = 0) in vec3 fragViewDir;
layout(location = 1) in vec3 fragNorm;
layout(location = 2) in vec2 fragTexCoord;
layout(location = 3) in vec3 fragPos;

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

/*
	compute spot light direction

	parameters:
		vec3 P: position of fragment
		vec3 LP: position of spot light
*/
vec3 spot_light_dir(vec3 P, vec3 LP) {

	vec3 dir = normalize(LP - P);

	return dir;
}

/*
	compute color of spot light

	parameters:
		vec3 P: position of fragment
		vec3 LP: position of spot light
		vec3 L : light direction
		vec3 C : light color
		float cIn : cosine of inner angle
		float cOut : cosine of outer angle
		float g : target distance
		float beta : decay factor

*/
vec3 spot_light_color(vec3 P, vec3 LP, vec3 L, vec3 C, float cIn, float cOut, float g, float beta) {

	float decay = g / length(LP - P);

	decay = pow(decay, beta);

	float clampValue = clamp( 
							(dot(normalize(LP - P), L) - cOut) /
							(cIn - cOut), 
							0, 1);

	vec3 color = C * decay * clampValue;

	return color;
}


void main() {
    //make sure this set of variables is the same for both day fragment shaders
    vec3 directLightDirection = normalize(vec3(0.2, 1.0, 0.2)); 
	vec3 directLightColor = vec3(0.1, 0.1, 0.1);
	vec3 carLightColor = vec3(1.0, 1.0, 1.0);
	float decayFactorSpotLight = 1.0;
	float cosineInnerSpotLight = cos(3.14 / 5);
	float cosineOuterSpotLight = cos(3.14 / 4);
	float targetDistanceSpotLight = 70;      //up to distance of max intensity
    vec3 ambientLight = vec3(0.01,0.01, 0.01);


	vec3  diffColor = texture(texSampler, fragTexCoord).rgb;
	float roughness = 1.0;
	
	vec3 normal = normalize(fragNorm);
	vec3 viewDir = normalize(fragViewDir);

	
	// Lambert diffuse
	vec3 diffuse  = Oren_Nayar_Diffuse_BRDF(directLightDirection, normal, viewDir, diffColor, roughness);

	vec3 color = directLightColor * (diffuse);

	diffuse = Oren_Nayar_Diffuse_BRDF(spot_light_dir(fragPos, gubo.rightFrontLightPos), 
									  normal, viewDir, diffColor, roughness);

	color += spot_light_color(fragPos,
							  gubo.rightFrontLightPos,
							  -gubo.carLightDir,
							  carLightColor,
							  cosineInnerSpotLight,
							  cosineOuterSpotLight,
							  targetDistanceSpotLight,
							  decayFactorSpotLight) *
			 diffuse;

	diffuse = Oren_Nayar_Diffuse_BRDF(spot_light_dir(fragPos, gubo.leftFrontLightPos), 
									  normal, viewDir, diffColor, roughness);

	color += spot_light_color(fragPos,
							  gubo.leftFrontLightPos,
							  -gubo.carLightDir,
							  carLightColor,
							  cosineInnerSpotLight,
							  cosineOuterSpotLight,
							  targetDistanceSpotLight,
							  decayFactorSpotLight) *
			 diffuse;

    diffuse = Oren_Nayar_Diffuse_BRDF(spot_light_dir(fragPos, gubo.rightRearLightPos), 
									  normal, viewDir, diffColor, roughness);

	color += spot_light_color(fragPos,
							  gubo.rightRearLightPos,
							  gubo.carLightDir,
							  gubo.backLightsColor,
							  cosineInnerSpotLight,
							  cosineOuterSpotLight,
							  targetDistanceSpotLight * 0.1,
							  decayFactorSpotLight + 1) *
			 diffuse;

    diffuse = Oren_Nayar_Diffuse_BRDF(spot_light_dir(fragPos, gubo.leftRearLightPos), 
									  normal, viewDir, diffColor, roughness);

	color += spot_light_color(fragPos,
							  gubo.leftRearLightPos,
							  gubo.carLightDir,
							  gubo.backLightsColor,
							  cosineInnerSpotLight,
							  cosineOuterSpotLight,
							  targetDistanceSpotLight * 0.1,
							  decayFactorSpotLight + 1) *
			 diffuse;

	// ambient lighting
	vec3 ambient  = ambientLight * diffColor;

	//adding ambient and clamping
	color = clamp((color + ambient), vec3(0.0, 0.0, 0.0), vec3(1.0, 1.0, 1.0));
	
	outColor = vec4(color, 1.0);
}