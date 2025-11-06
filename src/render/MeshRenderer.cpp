#include "MeshRenderer.h"

#include "GpuBuffers.h"
#include "platform/VulkanContext.h"
#include "platform/Swapchain.h"
#include <spdlog/spdlog.h>
#include <array>
#include <cstring>

namespace render {

namespace {
struct AvatarGeometry {
    std::array<MeshVertex, 8> vertices;
    std::array<uint32_t, 36> indices;
};

AvatarGeometry makeAvatarCube(float width, float height, float depth) {
    const float hw = width * 0.5f;
    const float hh = height;
    const float hd = depth * 0.5f;
    AvatarGeometry geo{};

    geo.vertices = {
        MeshVertex{{-hw, 0.0f, -hd}, {0.0f, 0.0f, -1.0f}},
        MeshVertex{{ hw, 0.0f, -hd}, {0.0f, 0.0f, -1.0f}},
        MeshVertex{{ hw, hh, -hd}, {0.0f, 0.0f, -1.0f}},
        MeshVertex{{-hw, hh, -hd}, {0.0f, 0.0f, -1.0f}},
        MeshVertex{{-hw, 0.0f,  hd}, {0.0f, 0.0f, 1.0f}},
        MeshVertex{{ hw, 0.0f,  hd}, {0.0f, 0.0f, 1.0f}},
        MeshVertex{{ hw, hh,  hd}, {0.0f, 0.0f, 1.0f}},
        MeshVertex{{-hw, hh,  hd}, {0.0f, 0.0f, 1.0f}},
    };

    geo.indices = {
        0, 1, 2, 2, 3, 0, // back
        4, 5, 6, 6, 7, 4, // front
        0, 4, 7, 7, 3, 0, // left
        1, 5, 6, 6, 2, 1, // right
        3, 2, 6, 6, 7, 3, // top
        0, 1, 5, 5, 4, 0  // bottom
    };
    return geo;
}

VkShaderModule createShader(VkDevice device, const std::vector<uint32_t>& code) {
    VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = code.size() * sizeof(uint32_t);
    ci.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }
    return module;
}

// A simple hard-coded SPIR-V for vertex/fragment would normally be loaded from disk.
// Placeholder binary will be filled later.
} // namespace

bool MeshRenderer::init(platform::VulkanContext& vk, platform::Swapchain& swap) {
    device_ = vk.device();
    physicalDevice_ = vk.physicalDevice();
    graphicsQueueFamily_ = vk.graphicsFamily();
    swapExtent_ = swap.extent();

    if (!createBuffers(vk)) return false;
    if (!createPipeline(vk, swap)) return false;

    ready_ = true;
    return true;
}

void MeshRenderer::shutdown(platform::VulkanContext& vk) {
    destroyPipeline(vk);
    destroyBuffers(vk);
    ready_ = false;
}

void MeshRenderer::resize(platform::VulkanContext& vk, platform::Swapchain& swap) {
    swapExtent_ = swap.extent();
    destroyPipeline(vk);
    createPipeline(vk, swap);
}

void MeshRenderer::updateInstances(const std::vector<MeshInstance>& instances) {
    instanceCount_ = instances.size();
    // TODO: upload to GPU
}

void MeshRenderer::record(VkCommandBuffer, uint32_t) {
    // TODO: record draw commands once pipeline is ready
}

bool MeshRenderer::createBuffers(platform::VulkanContext& vk) {
    AvatarGeometry geo = makeAvatarCube(0.6f, 2.0f, 0.4f);
    indexCount_ = static_cast<uint32_t>(geo.indices.size());
    // TODO: upload vertex/index to GPU buffers using existing upload helpers
    (void)vk;
    return true;
}

void MeshRenderer::destroyBuffers(platform::VulkanContext& vk) {
    if (vertexBuffer_) { vkDestroyBuffer(device_, vertexBuffer_, nullptr); vertexBuffer_ = VK_NULL_HANDLE; }
    if (vertexMemory_) { vkFreeMemory(device_, vertexMemory_, nullptr); vertexMemory_ = VK_NULL_HANDLE; }
    if (indexBuffer_) { vkDestroyBuffer(device_, indexBuffer_, nullptr); indexBuffer_ = VK_NULL_HANDLE; }
    if (indexMemory_) { vkFreeMemory(device_, indexMemory_, nullptr); indexMemory_ = VK_NULL_HANDLE; }
    if (instanceBuffer_) { vkDestroyBuffer(device_, instanceBuffer_, nullptr); instanceBuffer_ = VK_NULL_HANDLE; }
    if (instanceMemory_) { vkFreeMemory(device_, instanceMemory_, nullptr); instanceMemory_ = VK_NULL_HANDLE; }
    instanceCapacity_ = 0;
    instanceCount_ = 0;
}

bool MeshRenderer::createPipeline(platform::VulkanContext&, platform::Swapchain&) {
    // TODO: create pipeline
    return true;
}

void MeshRenderer::destroyPipeline(platform::VulkanContext&) {
    if (pipeline_) { vkDestroyPipeline(device_, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE; }
    if (pipelineLayout_) { vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE; }
    if (descLayout_) { vkDestroyDescriptorSetLayout(device_, descLayout_, nullptr); descLayout_ = VK_NULL_HANDLE; }
    if (descPool_) { vkDestroyDescriptorPool(device_, descPool_, nullptr); descPool_ = VK_NULL_HANDLE; }
    descSets_.clear();
}

bool MeshRenderer::ensureInstanceBuffer(size_t, platform::VulkanContext&) {
    return true;
}

} // namespace render

