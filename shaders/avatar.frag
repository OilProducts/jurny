#version 460

layout(location = 0) in vec3 vNormal;
layout(location = 1) in vec4 vColor;

layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CameraData {
    mat4 viewProj;
    vec3 lightDir;
    float _pad;
} pc;

void main() {
    vec3 N = normalize(vNormal);
    vec3 L = normalize(pc.lightDir);
    float diff = max(dot(N, L), 0.0);
    vec3 shading = vColor.rgb * (0.35 + 0.65 * diff);
    outColor = vec4(shading, 1.0);
}
