#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>

#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "shaders.h"

struct Vertex {
  glm::vec2 pos;
  glm::vec3 color;

  static vk::VertexInputBindingDescription getBindingDesc() {
    return {
      .binding = 0,
      .stride = sizeof(Vertex),
      .inputRate = vk::VertexInputRate::eVertex,
    };
  }

  static std::array<vk::VertexInputAttributeDescription, 2> getAttrsDescs() {
    return {
      vk::VertexInputAttributeDescription {
        .location = 0,
        .binding = 0,
        .format = vk::Format::eR32G32Sfloat,
        .offset = offsetof(Vertex, pos)
      },
      vk::VertexInputAttributeDescription {
        .location = 1,
        .binding = 0,
        .format = vk::Format::eR32G32B32Sfloat,
        .offset = offsetof(Vertex, color)
      },
    };
  }
};

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
  bool frameResized = false;

  static constexpr uint32_t WIDTH = 800;
  static constexpr uint32_t HEIGHT = 600;

  // frames in flight basically give an extra frame for the cpu to work on so
  // the cpu do other work while the gpu renders instead of waiting for the gpu
  // to finish first
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

  // info about what capabilities vulkan has on this system
  vk::raii::Context context;
  // provides available physical devices and surfaces
  vk::raii::Instance instance = nullptr;
  // allows debug messages from validation layers
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  // the window surface to draw on
  vk::raii::SurfaceKHR surface = nullptr;

  // represents the physical gpu
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  // an abstraction of a gpu
  vk::raii::Device device = nullptr;
  // a queue of commands and data to submit to the gpu
  vk::raii::Queue queue = nullptr;
  uint32_t queueIndex;

  // the queue of buffers the gpu renders to before presenting
  vk::raii::SwapchainKHR swapchain = nullptr;

  // the image buffers in the swapchain
  std::vector<vk::Image> swapchainImages;
  // how each image buffer should be interpreted as
  std::vector<vk::raii::ImageView> swapchainImageViews;
  // the format of the image buffers
  vk::Format swapchainFormat = vk::Format::eUndefined;
  // the size of the image buffers
  vk::Extent2D swapchainExtent;
  // which frame in flight we are rendering
  uint32_t frameIndex = 0;

  // lists the variables the cpu can set in the shaders
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  // the actual graphics pipeline
  vk::raii::Pipeline pipeline = nullptr;

  // the vertex buffer we will be render
  vk::raii::Buffer vertexBuffer = nullptr;
  // the actual memory for the vertex buffer
  vk::raii::DeviceMemory vertexBufferMem = nullptr;
  // the collection of command buffers for our queue
  vk::raii::CommandPool commandPool = nullptr;
  // one per frame in flight, the buffer to which we submit commands to the gpu
  std::vector<vk::raii::CommandBuffer> commandBuffers;

  // vulkan makes a distinction between rendering and presenting, so we need to
  // sync rendering and presenting to make sure there arent any funny
  // synchronisation errors

  // one per image view, blocks gpu until rendering is done
  std::vector<vk::raii::Semaphore> renderFinisheds;

  // one per frame in flight, blocks gpu until the buffer is presented
  std::vector<vk::raii::Semaphore> presentCompletes;

  // one per frame in flight, blocks cpu until a new draw command can be
  // submitted
  std::vector<vk::raii::Fence> drawFences;

  // triangle that we wanna render
  const std::vector<Vertex> vertices = {
    {{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}},
    {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}
  };

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
    createVertexBuffer();
    createCommandBuffers();
    createSyncObjects();
  }

  void createInstance() {
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

    // create instance
    constexpr vk::ApplicationInfo appInfo {
      .pApplicationName = "vulkan yay",
      .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
      .pEngineName = "No Engine",
      .engineVersion = VK_MAKE_VERSION(1, 0, 0),
      .apiVersion = vk::ApiVersion14,
    };

    vk::InstanceCreateInfo createInfo {
      .pApplicationInfo = &appInfo,
      .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
      .ppEnabledLayerNames = requiredLayers.data(),
      .enabledExtensionCount = static_cast<uint32_t>(requiredExts.size()),
      .ppEnabledExtensionNames = requiredExts.data(),
    };

    instance = vk::raii::Instance(context, createInfo);

    // list extensions
    std::cout << "available extensions:" << std::endl;
    for (const auto& extension : extProps) {
        std::cout << '\t' << extension.extensionName << '\n';
    }
  }

  std::vector<const char *> getRequiredExtensions() {
    // glfw requires certain extensions so we obtain those
    uint32_t extCount = 0;
    auto glfwExts = glfwGetRequiredInstanceExtensions(&extCount);
    std::vector extensions(glfwExts, glfwExts + extCount);

    // debug extension is required if we want validation layers
    if (enableValidationLayers)
      extensions.push_back(vk::EXTDebugUtilsExtensionName);

    // print out required extensions
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
    // just print the validation layer's debug message
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
    // some devices dont issue outofdate errors on window resize
    auto *app = reinterpret_cast<App *>(glfwGetWindowUserPointer(window));
    app->frameResized = true;
  }

  void pickPhysicalDevice() {
    auto devices = instance.enumeratePhysicalDevices();
    if (devices.empty()) {
      throw std::runtime_error("no vulkan compatible devices found! :(");
    }

    // print available devices
    std::cout << "available devices:" << std::endl;
    for (const auto &device : devices) {
      std::cout << '\t' << device.getProperties().deviceName << std::endl;
    }

    // we want a device that:
    // - supports vulkan api 1.3+
    // - supports a queue family with graphics and present capabilities
    // - has the features we require to draw
    for (const auto &device : devices) {
      // check vulkan api
      if (device.getProperties().apiVersion < VK_API_VERSION_1_3)
        continu: continue;

      // check queue families
      auto queueFamilies = device.getQueueFamilyProperties();
      auto indices = findQueueFamilies(device);
      if (!indices)
        continue;

      // check extensions
      auto extensions = device.enumerateDeviceExtensionProperties();
      for (auto const &ext : deviceExtensions) {
        if (std::ranges::none_of(extensions, [&](auto const &deviceExt) {
          return strcmp(ext, deviceExt.extensionName) == 0;
        })) goto continu;
      }

      // check features
      auto features = device.getFeatures2<vk::PhysicalDeviceFeatures2,
        vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
      if (!features.get<vk::PhysicalDeviceVulkan11Features>()
            .shaderDrawParameters ||
          !features.get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering ||
          !features.get<vk::PhysicalDeviceVulkan13Features>()
            .synchronization2 ||
          !features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState)
        continue;

      // print and set selected device
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
    // we want a queue family that can support both graphics and presenting to
    // a window surface
    auto queueFamilies = device.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++) {
      if (device.getSurfaceSupportKHR(i, surface) &&
        (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics))
        return i;
    }

    return std::nullopt;
  }

  void createLogicalDevice() {
    // set up queue
    auto qfp = physicalDevice.getQueueFamilyProperties();
    queueIndex = findQueueFamilies(physicalDevice).value();
    float queuePriority = 1;
    vk::DeviceQueueCreateInfo queueCreateInfo {
      .queueFamilyIndex = queueIndex,
      .queueCount = 1,
      .pQueuePriorities = &queuePriority
    };

    // set up device features
    vk::StructureChain<vk::PhysicalDeviceFeatures2,
      vk::PhysicalDeviceVulkan13Features,
      vk::PhysicalDeviceVulkan11Features,
      vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> features {
      { },
      { .synchronization2 = true, .dynamicRendering = true },
      { .shaderDrawParameters = true },
      { .extendedDynamicState = true }
    };

    // create device and queue
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
    // lots of picking options
    auto caps = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    auto availableFormats = physicalDevice.getSurfaceFormatsKHR(surface);
    auto format = chooseSurfaceFormat(availableFormats);
    auto availableModes = physicalDevice.getSurfacePresentModesKHR(surface);
    auto mode = choosePresentationMode(availableModes);
    auto extent = chooseSwapExtent(caps);

    // triple buffering is a nice default if supported, but go with whatever
    // the gpu can support
    uint32_t imageCount = std::max(3u, caps.minImageCount);
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
      imageCount = caps.maxImageCount;

    // create swap chain
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

    // we also want info about the swapchain images so we can create image
    // views and correctly render to the images
    swapchainImages = swapchain.getImages();
    swapchainFormat = format.format;
    swapchainExtent = extent;
  }

  vk::SurfaceFormatKHR chooseSurfaceFormat(
    const std::vector<vk::SurfaceFormatKHR> &availableFormats
  ) {
    // pick 32 bit bgra (srgb) if available, and pick the first option if not
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
    // pick mailbox if available, and fifo if not
    // the options are:
    // - immediate - present directly to the framebuffer with no intermediate
    //   buffer (prone to tearing)
    // - fifo - present from a queue of buffers, forcing the cpu to wait for
    //   the next available buffer in the queue if the queue is full
    // - fifo relaxed - like above, but if the queue is empty, present to the
    //   framebuffer instead
    // - mailbox - like fifo, but if the queue is full then replace the last
    //   entry
    for (auto mode : availableModes) {
      if (mode == vk::PresentModeKHR::eMailbox)
        return mode;
    }

    return vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR &caps) {
    // just get the size of the framebuffer lmao
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
    // each image is to be treated as a single 2d image
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
    // load shader module
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

    // describe vertex buffers
    auto bindingDesc = Vertex::getBindingDesc();
    auto attrDesc = Vertex::getAttrsDescs();
    vk::PipelineVertexInputStateCreateInfo vertexInputInfo {
      .vertexBindingDescriptionCount = 1,
      .pVertexBindingDescriptions = &bindingDesc,
      .vertexAttributeDescriptionCount = attrDesc.size(),
      .pVertexAttributeDescriptions = attrDesc.data()
    };

    // we want our viewport and scissor to be able to dynamically change so we
    // can resize the window (scissor is basically what portion of the viewport
    // is visible)
    std::vector dynamicStates = {
      vk::DynamicState::eViewport,
      vk::DynamicState::eScissor
    };

    vk::PipelineDynamicStateCreateInfo dynamicStateInfo {
      .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
      .pDynamicStates = dynamicStates.data()
    };

    vk::PipelineViewportStateCreateInfo viewportState {
      .viewportCount = 1,
      .scissorCount = 1
    };

    // how do we want our vertices to be assembled
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly {
      .topology = vk::PrimitiveTopology::eTriangleList,
    };

    // how do we want to rasterise our geometry
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

    // how do we want to treat multisampling
    vk::PipelineMultisampleStateCreateInfo multisampling {
      .rasterizationSamples = vk::SampleCountFlagBits::e1,
      .sampleShadingEnable = false
    };

    // how do we blend colours per swapchain
    vk::PipelineColorBlendAttachmentState colorBlendAttachment {
      .blendEnable = false,
      .colorWriteMask = vk::ColorComponentFlagBits::eR
                      | vk::ColorComponentFlagBits::eG
                      | vk::ColorComponentFlagBits::eB
                      | vk::ColorComponentFlagBits::eA
    };

    // how do we blend colours globally
    vk::PipelineColorBlendStateCreateInfo colorBlending {
      .logicOpEnable = false,
      .logicOp = vk::LogicOp::eCopy,
      .attachmentCount = 1,
      .pAttachments = &colorBlendAttachment
    };

    // what uniforms and push constants we can provide to the shaders
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo {
      .setLayoutCount = 0,
      .pushConstantRangeCount = 0
    };

    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    // what swapchains do we render with
    vk::PipelineRenderingCreateInfo renderingInfo {
      .colorAttachmentCount = 1,
      .pColorAttachmentFormats = &swapchainFormat
    };

    // put it all together to create the pipeline
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
    // our makefile compiles our shaders into a c++ file so we can just use
    // that for our shader module
    vk::ShaderModuleCreateInfo createInfo {
      .codeSize = shaders_len,
      .pCode = (const uint32_t *) &shaders,
    };

    return vk::raii::ShaderModule(device, createInfo);
  }

  void createVertexBuffer() {
    // create vertex buffer
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
    createBuffer(vertexBuffer, vertexBufferMem, bufferSize,
      vk::BufferUsageFlagBits::eVertexBuffer |
      vk::BufferUsageFlagBits::eTransferDst,
      vk::MemoryPropertyFlagBits::eDeviceLocal);

    // create staging buffer
    vk::raii::Buffer stagingBuffer = nullptr;
    vk::raii::DeviceMemory stagingBufferMem = nullptr;
    createBuffer(stagingBuffer, stagingBufferMem, bufferSize,
      vk::BufferUsageFlagBits::eTransferSrc,
      vk::MemoryPropertyFlagBits::eHostVisible |
      vk::MemoryPropertyFlagBits::eHostCoherent);

    // write our triangle to staging buffer
    void *data = stagingBufferMem.mapMemory(0, bufferSize);
    memcpy(data, vertices.data(), bufferSize);
    stagingBufferMem.unmapMemory();

    // copy staging buffer to vertex buffer
    copyBuffer(vertexBuffer, stagingBuffer, bufferSize);
  }

  void createBuffer(
    vk::raii::Buffer &buffer,
    vk::raii::DeviceMemory &mem,
    vk::DeviceSize size,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags props
  ) {
    // create buffer
    vk::BufferCreateInfo bufferInfo {
      .size = size,
      .usage = usage,
      .sharingMode = vk::SharingMode::eExclusive
    };

    buffer = vk::raii::Buffer(device, bufferInfo);

    // allocate buffer in memory
    vk::MemoryRequirements reqs = buffer.getMemoryRequirements();
    uint32_t memIndex = findMemoryType(reqs.memoryTypeBits, props);
    vk::MemoryAllocateInfo allocInfo {
      .allocationSize = reqs.size,
      .memoryTypeIndex = memIndex
    };

    mem = vk::raii::DeviceMemory(device, allocInfo);

    // bind the memory we just allocated to the buffer
    buffer.bindMemory(mem, 0);
  }

  void copyBuffer(
    vk::raii::Buffer &dst,
    vk::raii::Buffer &src,
    vk::DeviceSize size
  ) {
    vk::CommandBufferAllocateInfo commandInfo {
      .commandPool = commandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = 1,
    };

    auto copyCommandBuffer =
      std::move(vk::raii::CommandBuffers(device, commandInfo).front());
    copyCommandBuffer.begin(vk::CommandBufferBeginInfo {
      .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit
    });

    copyCommandBuffer.copyBuffer(src, dst, vk::BufferCopy(0, 0, size));
    copyCommandBuffer.end();

    queue.submit(vk::SubmitInfo {
      .commandBufferCount = 1,
      .pCommandBuffers = &*copyCommandBuffer
    }, nullptr);
    queue.waitIdle();
  }

  uint32_t findMemoryType(
    uint32_t typeFilter,
    vk::MemoryPropertyFlags propFilter
  ) {
    auto props = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < props.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
        (props.memoryTypes[i].propertyFlags & propFilter) == propFilter)
        return i;
    }

    throw std::runtime_error("could not find suitable device memory");
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo {
      .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
      .queueFamilyIndex = queueIndex
    };

    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  void createCommandBuffers() {
    // we want one command buffer per frame in flight
    vk::CommandBufferAllocateInfo allocInfo {
      .commandPool = commandPool,
      .level = vk::CommandBufferLevel::ePrimary,
      .commandBufferCount = MAX_FRAMES_IN_FLIGHT
    };

    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }

  void createSyncObjects() {
    // one render semaphore per imageview
    for (int i = 0; i < swapchainImages.size(); i++) {
      renderFinisheds.emplace_back(device, vk::SemaphoreCreateInfo {});
    }

    // one presentation semaphore and one draw fence per frame in flight
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

    // wait for the gpu to finish all commands before exiting
    device.waitIdle();
  }

  void drawFrame() {
    // the synchronisation will enforce (cpu -> render) -> presentation

    // wait on the cpu for the frame in flight to be available
    while (vk::Result::eTimeout == device.waitForFences(
      *drawFences[frameIndex], true, UINT64_MAX));

    // acquire the next image and signals the presentation semaphore when its
    // ready to render to
    auto [result, imageIndex] = swapchain.acquireNextImage(
      UINT64_MAX, *presentCompletes[frameIndex], nullptr);

    // check validity of swapchain
    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapchain();
      return;
    } else if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR)
      throw std::runtime_error("failed to acquire swapchain!");

    // reset fence for our frame in flight and draw
    device.resetFences(*drawFences[frameIndex]);
    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    vk::PipelineStageFlags waitDstStageMask{
      vk::PipelineStageFlagBits::eColorAttachmentOutput
    };

    // - submit a command on the queue that:
    //   - waits for the presentation semaphore
    //   - renders to the current buffer in the swapchain
    //   - signals the rendered semaphore
    //   - signals the draw fence
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
      // - submit a command on the queue that:
      //   - waits for the rendered semaphore
      //   - presents the current buffer in the swapchain to the framebuffer
      vk::PresentInfoKHR presentInfo {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &*renderFinisheds[imageIndex],
        .swapchainCount = 1,
        .pSwapchains = &*swapchain,
        .pImageIndices = &imageIndex
      };

      result = queue.presentKHR(presentInfo);

    // check validity of swapchain
      if (result == vk::Result::eErrorOutOfDateKHR
        || result == vk::Result::eSuboptimalKHR || frameResized) {
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

    // next frame in flight
    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    auto &commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});

    // prepare the image buffer for rendering colour to it
    transitionImageLayout(
      imageIndex,
      vk::ImageLayout::eUndefined,
      vk::ImageLayout::eColorAttachmentOptimal,
      {},
      vk::AccessFlagBits2::eColorAttachmentWrite,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput,
      vk::PipelineStageFlagBits2::eColorAttachmentOutput
    );

    // rendering settings relating to how to render
    vk::ClearValue clearColor = vk::ClearColorValue { 0.f, 0.f, 0.f, 1.f };
    vk::RenderingAttachmentInfo attachmentInfo {
      .imageView = swapchainImageViews[imageIndex],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = clearColor
    };

    // rendering settings relating to where to render to
    vk::RenderingInfo renderingInfo = {
      .renderArea = {
        .offset = { 0, 0 },
        .extent = swapchainExtent
      },
      .layerCount = 1,
      .colorAttachmentCount = 1,
      .pColorAttachments = &attachmentInfo
    };

    // our viewport is just the whole image
    vk::Viewport viewport {
      .x =  0.0f,
      .y = 0.0f,
      .width = static_cast<float>(swapchainExtent.width),
      .height = static_cast<float>(swapchainExtent.height),
      .minDepth = 0.0f,
      .maxDepth = 1.0f
    };

    // yippee yippee yay yay rendering!
    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
    commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
    commandBuffer.setViewport(0, viewport);
    commandBuffer.setScissor(0,
      vk::Rect2D(vk::Offset2D(0, 0), swapchainExtent));
    commandBuffer.draw(3, 1, 0, 0);
    commandBuffer.endRendering();

    // now we want the image buffer to be ready to present
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

  // this is used when the window is resized or minimised
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
