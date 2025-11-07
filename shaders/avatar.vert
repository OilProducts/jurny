#version 460

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in float inHeightRatio;
layout(location = 3) in vec4 inModel0;
layout(location = 4) in vec4 inModel1;
layout(location = 5) in vec4 inModel2;
layout(location = 6) in vec4 inModel3;
layout(location = 7) in vec4 inColor;

layout(location = 0) out vec3 vNormal;
layout(location = 1) out vec4 vColor;
layout(location = 2) out float vHeight;

layout(push_constant) uniform CameraData {
    mat4 viewProj;
    vec3 lightDir;
    float _pad;
} pc;

void main() {
    mat4 model = mat4(inModel0, inModel1, inModel2, inModel3);
    vec4 worldPos = model * vec4(inPosition, 1.0);
    gl_Position = pc.viewProj * worldPos;

    mat3 normalMat = mat3(model);
    vNormal = normalize(normalMat * inNormal);
    vColor = inColor;
    vHeight = clamp(inHeightRatio, 0.0, 1.0);
}
