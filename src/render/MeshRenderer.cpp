#include "MeshRenderer.h"

#include "core/Upload.h"
#include "platform/VulkanContext.h"
#include "platform/Swapchain.h"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <vector>

namespace render {

namespace {
struct AvatarGeometry {
    std::array<MeshVertex, 8> vertices;
    std::array<uint32_t, 36> indices;
};

AvatarGeometry makeAvatarCube(float width, float height, float depth) {
    const float hw = width * 0.5f;
    const float hd = depth * 0.5f;
    AvatarGeometry geo{};

    geo.vertices = {
        MeshVertex{{-hw, 0.0f, -hd}, { -1.0f,  0.0f,  0.0f }},
        MeshVertex{{ hw, 0.0f, -hd}, {  1.0f,  0.0f,  0.0f }},
        MeshVertex{{ hw, height, -hd}, {  1.0f,  0.0f,  0.0f }},
        MeshVertex{{-hw, height, -hd}, { -1.0f,  0.0f,  0.0f }},
        MeshVertex{{-hw, 0.0f,  hd}, { -1.0f,  0.0f,  0.0f }},
        MeshVertex{{ hw, 0.0f,  hd}, {  1.0f,  0.0f,  0.0f }},
        MeshVertex{{ hw, height,  hd}, {  1.0f,  0.0f,  0.0f }},
        MeshVertex{{-hw, height,  hd}, { -1.0f,  0.0f,  0.0f }},
    };

    geo.indices = {
        0, 1, 2, 2, 3, 0,
        4, 5, 6, 6, 7, 4,
        0, 4, 7, 7, 3, 0,
        1, 5, 6, 6, 2, 1,
        3, 2, 6, 6, 7, 3,
        0, 1, 5, 5, 4, 0
    };
    return geo;
}

std::vector<uint32_t> loadShaderBytes(const char* relPath) {
    const char* envAssets = std::getenv("VOXEL_ASSETS_DIR");
    const char* assetsDir = envAssets ? envAssets :
#ifdef VOXEL_ASSETS_DIR
        VOXEL_ASSETS_DIR;
#else
        "assets";
#endif
    std::string fullPath = std::string(assetsDir) + "/shaders/" + relPath;
    std::ifstream file(fullPath, std::ios::binary | std::ios::ate);
    if (!file) {
        spdlog::error("MeshRenderer failed to open shader {}", fullPath);
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if ((size % sizeof(uint32_t)) != 0) {
        spdlog::error("MeshRenderer shader {} has invalid size {}", fullPath, size);
        return {};
    }
    std::vector<uint32_t> data(static_cast<size_t>(size) / sizeof(uint32_t));
    if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
        spdlog::error("MeshRenderer failed to read shader {}", fullPath);
        return {};
    }
    return data;
}

VkShaderModule loadShaderModule(VkDevice device, const char* relPath) {
    std::vector<uint32_t> bytes = loadShaderBytes(relPath);
    if (bytes.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = bytes.size() * sizeof(uint32_t);
    ci.pCode = bytes.data();
    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &ci, nullptr, &module) != VK_SUCCESS) {
        spdlog::error("MeshRenderer failed to create shader module {}", relPath);
        return VK_NULL_HANDLE;
    }
    return module;
}
} // namespace

bool MeshRenderer::init(platform::VulkanContext& vk, platform::Swapchain& swap) {
    if (!vk.deviceInfo().hasDynamicRendering) {
        spdlog::warn("MeshRenderer disabled: dynamic rendering not supported");
        return false;
    }

    device_ = vk.device();
    physicalDevice_ = vk.physicalDevice();
    graphicsQueueFamily_ = vk.graphicsFamily();
    swapExtent_ = swap.extent();
    colorFormat_ = swap.format();

    if (!createBuffers(vk)) return false;
    if (!createPipeline(vk, swap)) return false;

    ready_ = true;
    return true;
}

void MeshRenderer::shutdown(platform::VulkanContext& vk) {
    if (!ready_) return;
    destroyPipeline(vk);
    destroyBuffers(vk);
    ready_ = false;
}

void MeshRenderer::resize(platform::VulkanContext& vk, platform::Swapchain& swap) {
    swapExtent_ = swap.extent();
    colorFormat_ = swap.format();
    destroyPipeline(vk);
    createPipeline(vk, swap);
}

void MeshRenderer::setCamera(const glm::mat4& viewProj, const glm::vec3& lightDir) {
    glm::vec3 normDir = glm::length(lightDir) > 1e-6f ? glm::normalize(lightDir) : glm::vec3(0.4f, 1.0f, 0.2f);
    pushConstants_.viewProj = viewProj;
    pushConstants_.lightDir = glm::vec4(normDir, 0.0f);
}

void MeshRenderer::updateInstances(const std::vector<MeshInstance>& instances) {
    instanceCount_ = instances.size();
    if (instances.empty()) {
        return;
    }
    if (!ensureInstanceBuffer(instances.size())) {
        instanceCount_ = 0;
        return;
    }
    std::memcpy(instanceMapped_, instances.data(), instances.size() * sizeof(MeshInstance));
}

void MeshRenderer::record(VkCommandBuffer cb, VkImage targetImage, VkImageView targetView) {
    if (!ready_ || instanceCount_ == 0 || pipeline_ == VK_NULL_HANDLE) return;
    if (vertexBuffer_ == VK_NULL_HANDLE || indexBuffer_ == VK_NULL_HANDLE || instanceBuffer_ == VK_NULL_HANDLE) return;

    VkImageMemoryBarrier toColor{};
    toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toColor.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toColor.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toColor.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toColor.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColor.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toColor.image = targetImage;
    toColor.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toColor.subresourceRange.levelCount = 1;
    toColor.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toColor);

    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = targetView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderingInfo.renderArea = {{0, 0}, swapExtent_};
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(cb, &renderingInfo);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(swapExtent_.width);
    viewport.height = static_cast<float>(swapExtent_.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = swapExtent_;
    vkCmdSetScissor(cb, 0, 1, &scissor);

    std::array<VkBuffer, 2> vbs{vertexBuffer_, instanceBuffer_};
    VkDeviceSize offsets[2] = {0, 0};
    vkCmdBindVertexBuffers(cb, 0, 2, vbs.data(), offsets);
    vkCmdBindIndexBuffer(cb, indexBuffer_, 0, VK_INDEX_TYPE_UINT32);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    vkCmdPushConstants(cb,
                       pipelineLayout_,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(PushConstants),
                       &pushConstants_);
    vkCmdDrawIndexed(cb, indexCount_, static_cast<uint32_t>(instanceCount_), 0, 0, 0);

    vkCmdEndRendering(cb);

    VkImageMemoryBarrier toGeneral{};
    toGeneral.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toGeneral.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    toGeneral.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toGeneral.image = targetImage;
    toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    toGeneral.subresourceRange.levelCount = 1;
    toGeneral.subresourceRange.layerCount = 1;
    vkCmdPipelineBarrier(cb,
                         VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0,
                         0, nullptr,
                         0, nullptr,
                         1, &toGeneral);
}

bool MeshRenderer::createBuffers(platform::VulkanContext& vk) {
    AvatarGeometry geo = makeAvatarCube(0.6f, 2.0f, 0.4f);
    indexCount_ = static_cast<uint32_t>(geo.indices.size());

    VkDeviceSize vertexBytes = sizeof(MeshVertex) * geo.vertices.size();
    VkDeviceSize indexBytes = sizeof(uint32_t) * geo.indices.size();

    if (!createBuffer(vertexBytes,
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      vertexBuffer_,
                      vertexMemory_)) {
        return false;
    }

    if (!createBuffer(indexBytes,
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                      indexBuffer_,
                      indexMemory_)) {
        return false;
    }

    core::UploadContext uploader;
    if (!uploader.init(vk)) {
        spdlog::error("MeshRenderer failed to initialise upload context");
        return false;
    }
    if (!uploader.uploadBuffer(geo.vertices.data(), vertexBytes, vertexBuffer_)) return false;
    if (!uploader.uploadBuffer(geo.indices.data(), indexBytes, indexBuffer_)) return false;
    uploader.flush();
    uploader.shutdown();

    return true;
}

void MeshRenderer::destroyBuffers(platform::VulkanContext&) {
    if (vertexBuffer_) { vkDestroyBuffer(device_, vertexBuffer_, nullptr); vertexBuffer_ = VK_NULL_HANDLE; }
    if (vertexMemory_) { vkFreeMemory(device_, vertexMemory_, nullptr); vertexMemory_ = VK_NULL_HANDLE; }
    if (indexBuffer_) { vkDestroyBuffer(device_, indexBuffer_, nullptr); indexBuffer_ = VK_NULL_HANDLE; }
    if (indexMemory_) { vkFreeMemory(device_, indexMemory_, nullptr); indexMemory_ = VK_NULL_HANDLE; }
    if (instanceBuffer_) { vkDestroyBuffer(device_, instanceBuffer_, nullptr); instanceBuffer_ = VK_NULL_HANDLE; }
    if (instanceMemory_) {
        if (instanceMapped_) {
            vkUnmapMemory(device_, instanceMemory_);
            instanceMapped_ = nullptr;
        }
        vkFreeMemory(device_, instanceMemory_, nullptr);
        instanceMemory_ = VK_NULL_HANDLE;
    }
    instanceCapacity_ = 0;
    instanceCount_ = 0;
}

bool MeshRenderer::createPipeline(platform::VulkanContext&, platform::Swapchain&) {
    VkShaderModule vs = loadShaderModule(device_, "avatar.vert.spv");
    VkShaderModule fs = loadShaderModule(device_, "avatar.frag.spv");
    if (!vs || !fs) {
        if (vs) vkDestroyShaderModule(device_, vs, nullptr);
        if (fs) vkDestroyShaderModule(device_, fs, nullptr);
        return false;
    }

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";

    VkVertexInputBindingDescription bindings[2]{};
    bindings[0].binding = 0;
    bindings[0].stride = sizeof(MeshVertex);
    bindings[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
    bindings[1].binding = 1;
    bindings[1].stride = sizeof(MeshInstance);
    bindings[1].inputRate = VK_VERTEX_INPUT_RATE_INSTANCE;

    VkVertexInputAttributeDescription attrs[7]{};
    attrs[0] = {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, position)};
    attrs[1] = {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(MeshVertex, normal)};
    attrs[2] = {2, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshInstance, model)};
    attrs[3] = {3, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshInstance, model) + sizeof(glm::vec4)};
    attrs[4] = {4, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshInstance, model) + sizeof(glm::vec4) * 2};
    attrs[5] = {5, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshInstance, model) + sizeof(glm::vec4) * 3};
    attrs[6] = {6, 1, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(MeshInstance, color)};

    VkPipelineVertexInputStateCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vi.vertexBindingDescriptionCount = 2;
    vi.pVertexBindingDescriptions = bindings;
    vi.vertexAttributeDescriptionCount = 7;
    vi.pVertexAttributeDescriptions = attrs;

    VkPipelineInputAssemblyStateCreateInfo ia{};
    ia.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo vp{};
    vp.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    vp.viewportCount = 1;
    vp.scissorCount = 1;

    VkDynamicState dynamicStates[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dyn{};
    dyn.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dyn.dynamicStateCount = 2;
    dyn.pDynamicStates = dynamicStates;

    VkPipelineRasterizationStateCreateInfo rs{};
    rs.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_BACK_BIT;
    rs.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rs.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo ms{};
    ms.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                           VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo cb{};
    cb.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    cb.attachmentCount = 1;
    cb.pAttachments = &blend;

    VkPushConstantRange pcr{};
    pcr.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pcr.offset = 0;
    pcr.size = sizeof(PushConstants);

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges = &pcr;
    if (vkCreatePipelineLayout(device_, &plci, nullptr, &pipelineLayout_) != VK_SUCCESS) {
        vkDestroyShaderModule(device_, vs, nullptr);
        vkDestroyShaderModule(device_, fs, nullptr);
        return false;
    }

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat_;

    VkGraphicsPipelineCreateInfo gpci{};
    gpci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    gpci.stageCount = 2;
    gpci.pStages = stages;
    gpci.pVertexInputState = &vi;
    gpci.pInputAssemblyState = &ia;
    gpci.pViewportState = &vp;
    gpci.pRasterizationState = &rs;
    gpci.pMultisampleState = &ms;
    gpci.pColorBlendState = &cb;
    gpci.pDynamicState = &dyn;
    gpci.layout = pipelineLayout_;
    gpci.pNext = &renderingInfo;

    VkResult res = vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1, &gpci, nullptr, &pipeline_);

    vkDestroyShaderModule(device_, vs, nullptr);
    vkDestroyShaderModule(device_, fs, nullptr);

    if (res != VK_SUCCESS) {
        spdlog::error("MeshRenderer failed to create graphics pipeline");
        return false;
    }
    return true;
}

void MeshRenderer::destroyPipeline(platform::VulkanContext&) {
    if (pipeline_) { vkDestroyPipeline(device_, pipeline_, nullptr); pipeline_ = VK_NULL_HANDLE; }
    if (pipelineLayout_) { vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr); pipelineLayout_ = VK_NULL_HANDLE; }
}

bool MeshRenderer::ensureInstanceBuffer(size_t count) {
    if (instanceCapacity_ >= count && instanceBuffer_) return true;

    if (instanceBuffer_) {
        vkDestroyBuffer(device_, instanceBuffer_, nullptr);
        instanceBuffer_ = VK_NULL_HANDLE;
    }
    if (instanceMemory_) {
        if (instanceMapped_) {
            vkUnmapMemory(device_, instanceMemory_);
            instanceMapped_ = nullptr;
        }
        vkFreeMemory(device_, instanceMemory_, nullptr);
        instanceMemory_ = VK_NULL_HANDLE;
    }

    instanceCapacity_ = std::max<size_t>(count, 4);
    VkDeviceSize bytes = sizeof(MeshInstance) * instanceCapacity_;
    if (!createBuffer(bytes,
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      instanceBuffer_,
                      instanceMemory_)) {
        instanceCapacity_ = 0;
        return false;
    }
    if (vkMapMemory(device_, instanceMemory_, 0, bytes, 0, &instanceMapped_) != VK_SUCCESS) {
        vkDestroyBuffer(device_, instanceBuffer_, nullptr);
        instanceBuffer_ = VK_NULL_HANDLE;
        vkFreeMemory(device_, instanceMemory_, nullptr);
        instanceMemory_ = VK_NULL_HANDLE;
        instanceCapacity_ = 0;
        return false;
    }
    return true;
}

bool MeshRenderer::createBuffer(VkDeviceSize size,
                                VkBufferUsageFlags usage,
                                VkMemoryPropertyFlags flags,
                                VkBuffer& buffer,
                                VkDeviceMemory& memory) {
    VkBufferCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    ci.size = size;
    ci.usage = usage;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &ci, nullptr, &buffer) != VK_SUCCESS) {
        spdlog::error("MeshRenderer failed to create buffer");
        return false;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, buffer, &req);
    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, flags);
    if (ai.memoryTypeIndex == UINT32_MAX) {
        spdlog::error("MeshRenderer could not match memory type");
        vkDestroyBuffer(device_, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    if (vkAllocateMemory(device_, &ai, nullptr, &memory) != VK_SUCCESS) {
        spdlog::error("MeshRenderer failed to allocate buffer memory");
        vkDestroyBuffer(device_, buffer, nullptr);
        buffer = VK_NULL_HANDLE;
        return false;
    }
    vkBindBufferMemory(device_, buffer, memory, 0);
    return true;
}

uint32_t MeshRenderer::findMemoryType(uint32_t bits, VkMemoryPropertyFlags requirements) const {
    VkPhysicalDeviceMemoryProperties memProps{};
    vkGetPhysicalDeviceMemoryProperties(physicalDevice_, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((bits & (1u << i)) &&
            (memProps.memoryTypes[i].propertyFlags & requirements) == requirements) {
            return i;
        }
    }
    return UINT32_MAX;
}

} // namespace render
