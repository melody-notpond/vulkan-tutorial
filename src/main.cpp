#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "shaders.h"

class App {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window;
  static constexpr uint32_t WIDTH = 800;
  static constexpr uint32_t HEIGHT = 600;
  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

  const std::vector<const char *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
  };

  const std::vector<const char *> deviceExtensions = {
    vk::KHRSwapchainExtensionName,
    vk::KHRSpirv14ExtensionName,
    vk::KHRSynchronization2ExtensionName,
    vk::KHRCreateRenderpass2ExtensionName
  };

  #ifdef NDEBUG
  static constexpr bool enableValidationLayers = false;
  #else
  static constexpr bool enableValidationLayers = true;
  #endif

  uint32_t queueIndex;

  std::vector<vk::Image> swapchainImages;
  vk::Format swapchainFormat = vk::Format::eUndefined;
  vk::Extent2D swapchainExtent;
  std::vector<vk::raii::ImageView> swapchainImageViews;
  uint32_t frameIndex = 0;
  bool frameResized = false;

  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swapchain = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline pipeline = nullptr;
  vk::raii::CommandPool commandPool = nullptr;

  std::vector<vk::raii::CommandBuffer> commandBuffers;
  std::vector<vk::raii::Semaphore> presentCompletes;
  std::vector<vk::raii::Semaphore> renderFinisheds;
  std::vector<vk::raii::Fence> drawFences;

  void initWindow() {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window = glfwCreateWindow(WIDTH, HEIGHT, "vulkan yay", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapchain();
    createImageViews();
    createGraphicsPipeline();
    createCommandPool();
    createCommandBuffers();
    createSyncObjects();
  }

  void createInstance() {
    constexpr vk::ApplicationInfo appInfo {
      .pApplicationName = "hewo wowd",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14,
    };

    // get required layers
    std::vector<const char *> requiredLayers;
    if (enableValidationLayers)
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());

    // check that all required layers are supported
    auto layerProps = context.enumerateInstanceLayerProperties();
    if (std::ranges::any_of(requiredLayers, [&](auto const &requiredLayer) {
        return std::ranges::none_of(layerProps, [&](auto const &layerProp) {
          return strcmp(layerProp.layerName, requiredLayer) == 0;
        });
    })) {
      throw std::runtime_error("some required layers are unsupported");
    }

    // get required extensions
    auto requiredExts = getRequiredExtensions();

    // check that all required extensions are supported
    auto extProps = context.enumerateInstanceExtensionProperties();
    if (std::ranges::any_of(requiredExts, [&](auto const &requiredExt) {
        return std::ranges::none_of(extProps, [&](auto const &extProp) {
          return strcmp(extProp.extensionName, requiredExt) == 0;
        });
    })) {
      throw std::runtime_error("some required extensions are unsupported");
    }

    vk::InstanceCreateInfo createInfo {
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
      .ppEnabledLayerNames = requiredLayers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(requiredExts.size()),
      .ppEnabledExtensionNames = requiredExts.data(),
    };

    // list extensions
    instance = vk::raii::Instance(context, createInfo);
    std::cout << "available extensions:" << std::endl;
    for (const auto& extension : extProps) {
        std::cout << '\t' << extension.extensionName << '\n';
    }
  }

  std::vector<const char *> getRequiredExtensions() {
    uint32_t extCount = 0;
    auto glfwExts = glfwGetRequiredInstanceExtensions(&extCount);

    std::vector extensions(glfwExts, glfwExts + extCount);
    if (enableValidationLayers)
      extensions.push_back(vk::EXTDebugUtilsExtensionName);

    std::cout << "required extensions:" << std::endl;
    for (const auto& extension : extensions) {
        std::cout << '\t' << extension << '\n';
    }

    return extensions;
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers)
      return;
    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
      // vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo    |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError 
    );
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes(
      vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral     |
      vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
      vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
    );
    vk::DebugUtilsMessengerCreateInfoEXT debugInfo {
      .messageSeverity = severityFlags,
      .messageType = messageTypes,
      .pfnUserCallback = &debugCallback
    };

    debugMessenger = instance.createDebugUtilsMessengerEXT(debugInfo);
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
    vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
    vk::DebugUtilsMessageTypeFlagsEXT type,
    const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData,
    void *pUserData
  ) {
    std::cerr << "validation layer: type " << to_string(type) << " msg: " <<
      pCallbackData->pMessage << std::endl;
    return vk::False;
  }

  void createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create surface");
    }

    surface = vk::raii::SurfaceKHR(instance, _surface);
  }

  static void framebufferResizeCallback(
    GLFWwindow *window,
    int width,
    int height
  ) {
    auto *app = reinterpret_cast<App *>(glfwGetWindowUserPointer(window));
    app->frameResized = true;
  }

  void pickPhysicalDevice() {
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
      throw std::runtime_error("no vulkan compatible devices found! :(");
    }

    std::cout << "available devices:" << std::endl;
    for (const auto &device : devices) {
      std::cout << '\t' << device.getProperties().deviceName << std::endl;
    }

    for (const auto &device : devices) {
      if (device.getProperties().apiVersion < VK_API_VERSION_1_3)
        continu: continue;

      auto queueFamilies = device.getQueueFamilyProperties();
      auto indices = findQueueFamilies(device);
      if (!indices)
        continue;

      auto extensions = device.enumerateDeviceExtensionProperties();
      for (auto const &ext : deviceExtensions) {
        if (std::ranges::none_of(extensions, [&](auto const &deviceExt) {
          return strcmp(ext, deviceExt.extensionName) == 0;
        })) goto continu;
      }

      auto features = device.getFeatures2<vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
      if (!features.get<vk::PhysicalDeviceVulkan11Features>()
            .shaderDrawParameters ||
          !features.get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering ||
          !features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState)
        continue;

      std::cout << "selected " << device.getProperties().deviceName <<
        std::endl;
      physicalDevice = device;
      return;
    }

    throw std::runtime_error("found no suitable vulkan device :(");
  }

  std::optional<uint32_t> findQueueFamilies(
    vk::raii::PhysicalDevice device
  ) {
    auto queueFamilies = device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
      if (device.getSurfaceSupportKHR(i, surface) &&
        (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics))
        return i;
    }

    return std::nullopt;
  }

  void createLogicalDevice() {
    auto qfp = physicalDevice.getQueueFamilyProperties();
    queueIndex = findQueueFamilies(physicalDevice).value();
    float queuePriority = 1;
    vk::DeviceQueueCreateInfo queueCreateInfo {
      .queueFamilyIndex = queueIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority
    };

    vk::StructureChain<vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceVulkan11Features,
      vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> features {
      { },
      { .synchronization2 = true, .dynamicRendering = true },
      { .shaderDrawParameters = true },
      { .extendedDynamicState = true }
    };

    vk::DeviceCreateInfo deviceCreateInfo {
      .pNext = features.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &queueCreateInfo,
      .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
      .ppEnabledExtensionNames = deviceExtensions.data()
    };

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    queue = vk::raii::Queue(device, queueIndex, 0);
  }

  void createSwapchain() {
    auto caps = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    auto availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
    auto format = chooseSurfaceFormat(availableFormats);
    auto availableModes = physicalDevice.getSurfacePresentModesKHR(surface);
    auto mode = choosePresentationMode(availableModes);
    auto extent = chooseSwapExtent(caps);

    uint32_t imageCount = std::max(3u, caps.minImageCount);
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
      imageCount = caps.maxImageCount;
    
    vk::SwapchainCreateInfoKHR createInfo {
      .flags = vk::SwapchainCreateFlagsKHR(),
      .surface = surface,
      .minImageCount = imageCount,
      .imageFormat = format.format,
      .imageColorSpace = format.colorSpace,
      .imageExtent = extent,
      .imageArrayLayers = 1,
      .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
      .imageSharingMode = vk::SharingMode::eExclusive,
      .queueFamilyIndexCount = 1,
      .pQueueFamilyIndices = &queueIndex,
      .preTransform = caps.currentTransform,
      .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
      .presentMode = mode,
      .clipped = vk::True,
      .oldSwapchain = nullptr
    };

    swapchain = vk::raii::SwapchainKHR(device, createInfo);
    swapchainImages = swapchain.getImages();
    swapchainFormat = format.format;
    swapchainExtent = extent;
  }

  vk::SurfaceFormatKHR chooseSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> &availableFormats
  ) {
    for (auto format : availableFormats) {
      if (format.format == vk::Format::eB8G8R8A8Srgb &&
        format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
        return format;
    }

    return availableFormats[0];
  }

  vk::PresentModeKHR choosePresentationMode(
    const std::vector<vk::PresentModeKHR> &availableModes
  ) {
    for (auto mode : availableModes) {
      if (mode == vk::PresentModeKHR::eMailbox)
        return mode;
    }

    return vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &caps) {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    return {
      std::clamp<uint32_t>(width, caps.minImageExtent.width,
        caps.maxImageExtent.width),
      std::clamp<uint32_t>(height, caps.minImageExtent.height,
        caps.maxImageExtent.height)
    };
  }

  void createImageViews() {
    vk::ImageViewCreateInfo createInfo {
      .viewType = vk::ImageViewType::e2D,
      .format = swapchainFormat,
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };

    for (auto image : swapchainImages) {
      createInfo.image = image;
      swapchainImageViews.emplace_back(device, createInfo);
    }
  }

  void createGraphicsPipeline() {
    auto shaders = createShaderModule();
    vk::PipelineShaderStageCreateInfo vertShaderInfo {
      .stage = vk::ShaderStageFlagBits::eVertex,
      .module = shaders,
      .pName = "vert_main"
    };

    vk::PipelineShaderStageCreateInfo fragShaderInfo {
      .stage = vk::ShaderStageFlagBits::eFragment,
      .module = shaders,
      .pName = "frag_main"
    };

    vk::PipelineShaderStageCreateInfo shaderStages[] {
      vertShaderInfo, fragShaderInfo
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};

    std::vector dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicStateInfo {
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()
    };

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly {
      .topology = vk::PrimitiveTopology::eTriangleList
    };

    vk::PipelineViewportStateCreateInfo viewportState {
      .viewportCount = 1,
      .scissorCount = 1
    };

    vk::PipelineRasterizationStateCreateInfo rasteriser {
      .depthClampEnable = false,
      .rasterizerDiscardEnable = false,
      .polygonMode = vk::PolygonMode::eFill,
      .cullMode = vk::CullModeFlagBits::eBack,
      .frontFace = vk::FrontFace::eClockwise,
      .depthBiasEnable = false,
      .depthBiasSlopeFactor = 1.0,
      .lineWidth = 1.0
    };

    vk::PipelineMultisampleStateCreateInfo multisampling {
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = false
    };

    vk::PipelineColorBlendAttachmentState colorBlendAttachment {
      .blendEnable = false,
      .colorWriteMask = vk::ColorComponentFlagBits::eR
                      | vk::ColorComponentFlagBits::eG
                      | vk::ColorComponentFlagBits::eB
                      | vk::ColorComponentFlagBits::eA
    };

    vk::PipelineColorBlendStateCreateInfo colorBlending {
      .logicOpEnable = false,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment
    };

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
      .setLayoutCount = 0,
      .pushConstantRangeCount = 0
    };

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::PipelineRenderingCreateInfo renderingInfo {
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapchainFormat
    };
    vk::GraphicsPipelineCreateInfo pipelineInfo {
      .pNext = &renderingInfo,
      .stageCount = 2,
      .pStages = shaderStages,
      .pVertexInputState = &vertexInputInfo,
      .pInputAssemblyState = &inputAssembly,
      .pViewportState = &viewportState,
      .pRasterizationState = &rasteriser,
      .pMultisampleState = &multisampling,
      .pColorBlendState = &colorBlending,
      .pDynamicState = &dynamicStateInfo,
      .layout = pipelineLayout,
      .renderPass = nullptr
    };

    pipeline = vk::raii::Pipeline(device, nullptr, pipelineInfo);
  }

  [[nodiscard]]
  vk::raii::ShaderModule createShaderModule() {
    vk::ShaderModuleCreateInfo createInfo {
      .codeSize = shaders_len,
      .pCode = (const uint32_t *) &shaders,
    };

    return vk::raii::ShaderModule(device, createInfo);
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = queueIndex
    };

    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  void createCommandBuffers() {
    vk::CommandBufferAllocateInfo allocInfo {
      .commandPool = commandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };

    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }

  void createSyncObjects() {
    for (int i = 0; i < swapchainImages.size(); i++) {
      renderFinisheds.emplace_back(device, vk::SemaphoreCreateInfo {});
    }

    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      presentCompletes.emplace_back(device, vk::SemaphoreCreateInfo {});
      drawFences.emplace_back(device, vk::FenceCreateInfo {
        .flags = vk::FenceCreateFlagBits::eSignaled
      });
    }
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }

    device.waitIdle();
  }

  void drawFrame() {
    while (vk::Result::eTimeout == device.waitForFences(
      *drawFences[frameIndex], true, UINT64_MAX));

    auto [result, imageIndex] = swapchain.acquireNextImage(
      UINT64_MAX, *presentCompletes[frameIndex], nullptr);

    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapchain();
      return;
    } else if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("failed to acquire swapchain!");

    device.resetFences(*drawFences[frameIndex]);
    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDstStageMask{
      vk::PipelineStageFlagBits::eColorAttachmentOutput
    };

    vk::SubmitInfo submitInfo {
      .waitSemaphoreCount = 1,
      .pWaitSemaphores = &*presentCompletes[frameIndex],
      .pWaitDstStageMask = &waitDstStageMask,
      .commandBufferCount = 1,
      .pCommandBuffers = &*commandBuffers[frameIndex],
      .signalSemaphoreCount = 1,
      .pSignalSemaphores = &*renderFinisheds[imageIndex]
    };

    queue.submit(submitInfo, drawFences[frameIndex]);

    try {
      vk::PresentInfoKHR presentInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinisheds[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain,
        .pImageIndices = &imageIndex
      };

      result = queue.presentKHR(presentInfo);
      if (result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || frameResized) {
        frameResized = false;
        recreateSwapchain();
      } else if (result != vk::Result::eSuccess)
         throw std::runtime_error("could not present to swapchain image");
    } catch (const vk::SystemError &e) {
      if (e.code().value() == static_cast<int>(vk::Result::eErrorOutOfDateKHR))
      {
        recreateSwapchain();
        return;
      } else throw;
    }

    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    auto &commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});
    transitionImageLayout(
      imageIndex,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal,
      {},
      vk::AccessFlagBits2::eColorAttachmentWrite,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput
    );

    vk::ClearValue clearColor = vk::ClearColorValue { 0.f, 0.f, 0.f, 1.f };
    vk::RenderingAttachmentInfo attachmentInfo {
      .imageView = swapchainImageViews[imageIndex],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clearColor
    };

    vk::RenderingInfo renderingInfo = {
      .renderArea = {
        .offset = { 0, 0 },
        .extent = swapchainExtent
      },
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachmentInfo
    };

    vk::Viewport viewport {
      .x =  0.0f,
      .y = 0.0f,
      .width = static_cast<float>(swapchainExtent.width),
      .height = static_cast<float>(swapchainExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f
    };

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline);
    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0,
      vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    transitionImageLayout(
      imageIndex,
      vk::ImageLayout::eColorAttachmentOptimal,
      vk::ImageLayout::ePresentSrcKHR,
      vk::AccessFlagBits2::eColorAttachmentWrite,
      {},
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      vk::PipelineStageFlagBits2::eBottomOfPipe
    );

    commandBuffer.end();
  }

  void transitionImageLayout(
    uint32_t imageIndex,
    vk::ImageLayout oldLayout,
    vk::ImageLayout newLayout,
    vk::AccessFlags2 srcAccessMask,
    vk::AccessFlags2 dstAccessMask,
    vk::PipelineStageFlags2 srcStageMask,
    vk::PipelineStageFlags2 dstStageMask
  ) {
    vk::ImageMemoryBarrier2 barrier {
      .srcStageMask = srcStageMask,
      .srcAccessMask = srcAccessMask,
      .dstStageMask = dstStageMask,
      .dstAccessMask = dstAccessMask,
      .oldLayout = oldLayout,
      .newLayout = newLayout,
      .srcQueueFamilyIndex = vk::QueueFamilyIgnored,
      .dstQueueFamilyIndex = vk::QueueFamilyIgnored,
      .image = swapchainImages[imageIndex],
      .subresourceRange = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .baseMipLevel = 0,
        .levelCount = 1,
        .baseArrayLayer = 0,
        .layerCount = 1
      }
    };

    vk::DependencyInfo dependencyInfo {
      .dependencyFlags = {},
      .imageMemoryBarrierCount = 1,
      .pImageMemoryBarriers = &barrier
    };

    commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
  }

  void recreateSwapchain() {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwWaitEvents();
      glfwGetFramebufferSize(window, &width, &height);
    }

    device.waitIdle();

    cleanupSwapchain();
    createSwapchain();
    createImageViews();
  }

  void cleanupSwapchain() {
    swapchainImageViews.clear();
    swapchain = nullptr;
  }

  void cleanup() {
    cleanupSwapchain();
    surface = nullptr;
    glfwDestroyWindow(window);
    glfwTerminate();
  }
};

int main(void) {
  App app{};

  try {
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
