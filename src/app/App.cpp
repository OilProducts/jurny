#include "App.h"
#include "platform/VulkanContext.h"
#if VOXEL_ENABLE_WINDOW
#include "platform/Window.h"
#include "platform/Swapchain.h"
#endif

#include <volk.h>
#include <vector>

namespace app {
int App::run() {
    platform::VulkanContext vk;
    bool enableValidation = true;
#if VOXEL_ENABLE_WINDOW
    platform::Window window;
    if (!window.create()) return 1;
    std::vector<const char*> instanceExts;
    platform::Window::getRequiredInstanceExtensions(instanceExts);
    if (!vk.initInstance(instanceExts, enableValidation)) return 1;
    VkSurfaceKHR surface = (VkSurfaceKHR)0;
    if (!window.createSurface(vk.instance(), &surface)) return 1;
    if (!vk.initDevice(surface)) return 1;
    platform::Swapchain swap;
    platform::SwapchainCreateInfo sci{};
    sci.device = vk.device();
    sci.physicalDevice = vk.physicalDevice();
    sci.surface = surface;
    sci.graphicsQueueFamily = vk.graphicsFamily();
    sci.presentQueueFamily = vk.graphicsFamily();
    sci.width = window.width();
    sci.height= window.height();
    swap.create(sci);

    // First-pixels: create a tiny compute pipeline to write a gradient
    VkDescriptorSetLayoutBinding b0{}; b0.binding = 0; b0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; b0.descriptorCount = 1; b0.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    VkDescriptorSetLayoutCreateInfo dslci{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    dslci.bindingCount = 1; dslci.pBindings = &b0;
    VkDescriptorSetLayout dsl{}; vkCreateDescriptorSetLayout(vk.device(), &dslci, nullptr, &dsl);
    VkPipelineLayoutCreateInfo plci{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plci.setLayoutCount = 1; plci.pSetLayouts = &dsl;
    VkPipelineLayout pl{}; vkCreatePipelineLayout(vk.device(), &plci, nullptr, &pl);

    auto readFile = [](const char* path) -> std::vector<uint32_t> {
        std::vector<uint32_t> out; FILE* f = fopen(path, "rb"); if (!f) return out; fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET); out.resize((sz+3)/4); fread(out.data(), 1, sz, f); fclose(f); return out; };
    std::vector<uint32_t> spv = readFile("assets/shaders/first_pixels.comp.spv");
    if (spv.empty()) spv = readFile("build/shaders/first_pixels.comp.spv");
    VkShaderModuleCreateInfo smci{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smci.codeSize = spv.size()*sizeof(uint32_t); smci.pCode = spv.data();
    VkShaderModule sm{}; vkCreateShaderModule(vk.device(), &smci, nullptr, &sm);
    VkComputePipelineCreateInfo cpci{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    VkPipelineShaderStageCreateInfo ssci{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
    ssci.stage = VK_SHADER_STAGE_COMPUTE_BIT; ssci.module = sm; ssci.pName = "main"; cpci.stage = ssci; cpci.layout = pl;
    VkPipeline pipe{}; vkCreateComputePipelines(vk.device(), VK_NULL_HANDLE, 1, &cpci, nullptr, &pipe);

    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, swap.imageCount() };
    VkDescriptorPoolCreateInfo dpci{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    dpci.maxSets = swap.imageCount(); dpci.poolSizeCount = 1; dpci.pPoolSizes = &poolSize;
    VkDescriptorPool dpool{}; vkCreateDescriptorPool(vk.device(), &dpci, nullptr, &dpool);
    std::vector<VkDescriptorSet> sets(swap.imageCount()); std::vector<VkDescriptorSetLayout> layouts(swap.imageCount(), dsl);
    VkDescriptorSetAllocateInfo dsai{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = swap.imageCount(); dsai.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(vk.device(), &dsai, sets.data());
    for (uint32_t i=0;i<swap.imageCount();++i) { VkDescriptorImageInfo di{}; di.imageView = swap.imageViews()[i]; di.imageLayout = VK_IMAGE_LAYOUT_GENERAL; VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET }; w.dstSet=sets[i]; w.dstBinding=0; w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w.descriptorCount=1; w.pImageInfo=&di; vkUpdateDescriptorSets(vk.device(),1,&w,0,nullptr); }

    VkCommandBuffer cb{}; VkCommandBufferAllocateInfo cbai{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO }; cbai.commandPool = vk.commandPool(); cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1; vkAllocateCommandBuffers(vk.device(), &cbai, &cb);
    VkSemaphoreCreateInfo sciSem{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO }; VkSemaphore acquireSem{}, finishSem{}; vkCreateSemaphore(vk.device(), &sciSem, nullptr, &acquireSem); vkCreateSemaphore(vk.device(), &sciSem, nullptr, &finishSem);
    while (!window.shouldClose()) {
        window.poll(); uint32_t idx=0; if (vkAcquireNextImageKHR(vk.device(), swap.handle(), UINT64_MAX, acquireSem, VK_NULL_HANDLE, &idx) != VK_SUCCESS) break;
        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO }; vkBeginCommandBuffer(cb, &bi);
        VkImageMemoryBarrier toGeneral{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        toGeneral.srcAccessMask = 0; toGeneral.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT; toGeneral.oldLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; toGeneral.newLayout = VK_IMAGE_LAYOUT_GENERAL; toGeneral.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toGeneral.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toGeneral.image = swap.image(idx); toGeneral.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; toGeneral.subresourceRange.levelCount = 1; toGeneral.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0,nullptr, 0,nullptr, 1, &toGeneral);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &sets[idx], 0, nullptr);
        uint32_t gx = (swap.extent().width + 7u)/8u, gy = (swap.extent().height + 7u)/8u; vkCmdDispatch(cb, gx, gy, 1);
        VkImageMemoryBarrier toPresent{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        toPresent.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; toPresent.dstAccessMask = 0; toPresent.oldLayout = VK_IMAGE_LAYOUT_GENERAL; toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR; toPresent.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toPresent.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED; toPresent.image = swap.image(idx); toPresent.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT; toPresent.subresourceRange.levelCount = 1; toPresent.subresourceRange.layerCount = 1;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,nullptr, 0,nullptr, 1, &toPresent);
        vkEndCommandBuffer(cb);
        VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT; VkSubmitInfo si{ VK_STRUCTURE_TYPE_SUBMIT_INFO }; si.waitSemaphoreCount=1; si.pWaitSemaphores=&acquireSem; si.pWaitDstStageMask=&waitStage; si.commandBufferCount=1; si.pCommandBuffers=&cb; si.signalSemaphoreCount=1; si.pSignalSemaphores=&finishSem; vkQueueSubmit(vk.graphicsQueue(), 1, &si, VK_NULL_HANDLE);
        VkPresentInfoKHR pi{ VK_STRUCTURE_TYPE_PRESENT_INFO_KHR }; VkSwapchainKHR sc=swap.handle(); pi.waitSemaphoreCount=1; pi.pWaitSemaphores=&finishSem; pi.swapchainCount=1; pi.pSwapchains=&sc; pi.pImageIndices=&idx; vkQueuePresentKHR(vk.graphicsQueue(), &pi); vkQueueWaitIdle(vk.graphicsQueue());
    }

    vkDestroySemaphore(vk.device(), finishSem, nullptr); vkDestroySemaphore(vk.device(), acquireSem, nullptr);
    vkDestroyPipeline(vk.device(), pipe, nullptr); vkDestroyShaderModule(vk.device(), sm, nullptr);
    vkDestroyDescriptorPool(vk.device(), dpool, nullptr); vkDestroyPipelineLayout(vk.device(), pl, nullptr); vkDestroyDescriptorSetLayout(vk.device(), dsl, nullptr);
    swap.destroy(vk.device());
    vk.shutdown();
    window.destroy();
#else
    // Headless: no window, no surface, no swapchain
    if (!vk.initInstance({}, enableValidation)) return 1;
    if (!vk.initDevice(VK_NULL_HANDLE)) return 1; // device without swapchain extension
    // Do a trivial no-op to ensure init path runs
    vk.shutdown();
#endif
    return 0;
}
}
