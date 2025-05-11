//! Graphics context definitions and setup routines, including physical device selection.

use crate::gfx::consts::{
    EXTENSION_VALIDATION_LAYER, PORTABILITY_MACOS_VERSION, VALIDATION_ENABLED,
};
use crate::gfx::debug::logging::debug_callback;
use crate::gfx::pipeline::AppPipeline;
use crate::gfx::queuefamily::QueueFamilyIndices;
use crate::gfx::swapchain::{SwapchainSupport, create_swapchain, create_swapchain_image_views};
use anyhow::Result;
use anyhow::anyhow;
use log::{debug, info, warn};
use std::collections::HashSet;
use std::ffi::c_char;
use thiserror::Error;
use vulkanalia::loader::LibloadingLoader;
use vulkanalia::vk::{
    DeviceV1_0, EntryV1_0, ExtDebugUtilsExtension, HasBuilder, InstanceV1_0, KhrSurfaceExtension,
    KhrSwapchainExtension,
};
use vulkanalia::{Device, Entry, Instance, vk, window as vk_window};
use vulkanalia_sys::{Handle, InstanceCreateFlags};
use winit::window::Window;

pub struct AppExtensionConfig {
    extensions: Vec<*const c_char>,
    flags: InstanceCreateFlags,
    layers: Vec<*const c_char>,
}

const MAX_FRAMES_IN_FLIGHT: usize = 2;

unsafe fn create_instance_info(
    window: &Window,
    entry: &Entry,
) -> anyhow::Result<AppExtensionConfig> {
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    let available_layers = unsafe {
        entry
            .enumerate_instance_layer_properties()?
            .iter()
            .map(|l| l.layer_name)
            .collect::<HashSet<_>>()
    };

    if VALIDATION_ENABLED && !available_layers.contains(&EXTENSION_VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![EXTENSION_VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    if VALIDATION_ENABLED {
        extensions.push(vulkanalia_sys::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(
            vulkanalia_sys::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(
            vulkanalia_sys::KHR_PORTABILITY_ENUMERATION_EXTENSION
                .name
                .as_ptr(),
        );
        InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        InstanceCreateFlags::empty()
    };

    Ok(AppExtensionConfig {
        extensions,
        flags,
        layers,
    })
}

pub unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut AppData,
) -> anyhow::Result<Instance> {
    unsafe {
        let application_info = vulkanalia_sys::ApplicationInfo::builder()
            .application_name(b"gfx :3\0")
            .application_version(vulkanalia_sys::make_version(0, 1, 0))
            .engine_name(b"nya\0")
            .engine_version(vulkanalia_sys::make_version(0, 1, 0))
            .api_version(vulkanalia_sys::make_version(1, 0, 0));

        let config = create_instance_info(window, entry)?;

        let mut info = vulkanalia_sys::InstanceCreateInfo::builder()
            .application_info(&application_info)
            .enabled_layer_names(&config.layers)
            .enabled_extension_names(&config.extensions)
            .flags(config.flags);

        let mut debug_info = vulkanalia_sys::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(vulkanalia_sys::DebugUtilsMessageSeverityFlagsEXT::all())
            .message_type(
                vulkanalia_sys::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vulkanalia_sys::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vulkanalia_sys::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .user_callback(Some(debug_callback));

        if VALIDATION_ENABLED {
            debug!("Vulkan validation layers are enabled.");
            info = info.push_next(&mut debug_info);
        }

        let instance = entry.create_instance(&info, None)?;
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;

        Ok(instance)
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

unsafe fn check_physical_device(
    instance: &Instance,
    data: &AppData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    unsafe {
        QueueFamilyIndices::get(instance, data.surface, physical_device)?;

        check_physical_device_extensions(instance, physical_device)?;

        let swapchain_support = SwapchainSupport::get(&instance, &data, physical_device)?;
        if swapchain_support.formats.is_empty() || swapchain_support.present_modes.is_empty() {
            Err(anyhow!(SuitabilityError("No swapchain support found.")))
        } else {
            Ok(())
        }
    }
}

const REQUIRED_DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    unsafe {
        let extensions = instance
            .enumerate_device_extension_properties(physical_device, None)?
            .iter()
            .map(|e| e.extension_name)
            .collect::<HashSet<_>>();

        if REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .all(|e| extensions.contains(e))
        {
            Ok(())
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required device extensions."
            )))
        }
    }
}

unsafe fn pick_physical_device(instance: &Instance, data: &mut AppData) -> Result<()> {
    unsafe {
        for physical_device in instance.enumerate_physical_devices()? {
            let properties = instance.get_physical_device_properties(physical_device);

            if let Err(error) = check_physical_device(instance, data, physical_device) {
                warn!(
                    "Skipping physical device (`{}`): {}",
                    properties.device_name, error
                );
            } else {
                info!("Selected physical device (`{}`).", properties.device_name);
                data.physical_device = physical_device;
                return Ok(());
            }
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut AppData,
) -> Result<Device> {
    unsafe {
        let indices = QueueFamilyIndices::get(instance, data.surface, data.physical_device)?;

        let mut queried_indices = HashSet::new();
        queried_indices.insert(indices.graphics);
        queried_indices.insert(indices.present);

        let queue_priorities = &[1.0];
        let queue_infos = queried_indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();

        let mut layers: Vec<*const c_char> = vec![];
        if VALIDATION_ENABLED {
            layers.push(EXTENSION_VALIDATION_LAYER.as_ptr());
        }

        // as C strings
        let mut extensions = REQUIRED_DEVICE_EXTENSIONS
            .iter()
            .map(|n| n.as_ptr())
            .collect::<Vec<_>>();

        // Required by Vulkan SDK on macOS since 1.3.216.
        if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
            extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
        }

        let features = vk::PhysicalDeviceFeatures::builder();

        let info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = instance.create_device(data.physical_device, &info, None)?;

        data.graphics_queue = device.get_device_queue(indices.graphics, 0);
        data.present_queue = device.get_device_queue(indices.present, 0);

        Ok(device)
    }
}

#[derive(Clone, Debug)]
pub struct App {
    entry: Entry,
    instance: Instance,
    data: AppData,
    device: Device,
    frame_in_flight_idx: usize,
    pub(crate) resized: bool,
}

impl App {
    pub unsafe fn create(window: &Window) -> anyhow::Result<Self> {
        unsafe {
            let loader = LibloadingLoader::new(vulkanalia::loader::LIBRARY)?;
            let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
            let mut data = AppData::default();
            let instance = create_instance(window, &entry, &mut data)?;

            // the surface must be created prior to picking the physical device
            data.surface = vk_window::create_surface(&instance, &window, &window)?;

            pick_physical_device(&instance, &mut data)?;
            let device = create_logical_device(&entry, &instance, &mut data)?;

            create_swapchain(&window, &instance, &device, &mut data)?;
            info!(
                "Swapchain acquired ({} images, {:?}, {:?})",
                data.swapchain_images.len(),
                data.swapchain_format,
                data.swapchain_extent
            );

            create_swapchain_image_views(&device, &mut data)?;

            create_render_pass(&instance, &device, &mut data)?;
            data.render_pipeline =
                AppPipeline::new(&device, &data.swapchain_extent, data.main_render_pass)?;

            create_framebuffers(&device, &mut data)?;

            create_command_pool(&instance, &device, &mut data)?;
            create_command_buffers(&device, &mut data)?;
            create_sync_objects(&device, &mut data)?;

            Ok(Self {
                entry,
                instance,
                data,
                device,
                frame_in_flight_idx: 0,
                resized: false,
            })
        }
    }

    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        unsafe {
            self.device.device_wait_idle()?;
            self.destroy_swapchain_and_pipeline();

            // TODO: move swapchain fields and stuff to a unique struct
            // (maybe after dynamic rendering to reduce dependencies to the swapchain?)
            // part of the issue is that recreating the chain isn't strictly the same task as destroying
            // and recreating it, e.g. because we can opt to just clear the command buffers

            create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
            info!(
                "Swapchain acquired ({} images, {:?}, {:?})",
                self.data.swapchain_images.len(),
                self.data.swapchain_format,
                self.data.swapchain_extent
            );
            create_swapchain_image_views(&self.device, &mut self.data)?;
            create_render_pass(&self.instance, &self.device, &mut self.data)?;
            self.data.render_pipeline = AppPipeline::new(
                &self.device,
                &self.data.swapchain_extent,
                self.data.main_render_pass,
            )?;
            create_framebuffers(&self.device, &mut self.data)?;
            create_command_buffers(&self.device, &mut self.data)?;
            self.data
                .in_flight_swapchain_image_fences
                .resize(self.data.swapchain_images.len(), vk::Fence::null());
            Ok(())
        }
    }

    /// Destroy the swapchain, render passes and rendering pipelines, which attach to the
    /// framebuffers.
    ///
    /// # Safety
    ///
    /// The caller must ensure that the swapchain is not actually being used on the device, e.g.
    /// by using `device.device_wait_idle().unwrap();`.
    ///
    /// The caller must ensure that swapchain resources are not subsequently used by other resources
    /// (e.g. command buffers that bind swapchain resources).
    unsafe fn destroy_swapchain_and_pipeline(&mut self) {
        unsafe {
            self.data
                .framebuffers
                .iter()
                .for_each(|f| self.device.destroy_framebuffer(*f, None));

            self.device.free_command_buffers(
                self.data.command_pool,
                &self.data.command_buffers
            );

            self.data.render_pipeline.destroy(&self.device);

            self.device.destroy_render_pass(self.data.main_render_pass, None);

            self.data
                .swapchain_image_views
                .iter()
                .for_each(|v| self.device.destroy_image_view(*v, None));

            self.device.destroy_swapchain_khr(self.data.swapchain, None);
        }
    }

    pub unsafe fn render(&mut self, window: &Window) -> Result<()> {
        if !self.data.in_flight_swapchain_image_fences[self.frame_in_flight_idx as usize].is_null()
        {
            self.device.wait_for_fences(
                &[self.data.in_flight_swapchain_image_fences[self.frame_in_flight_idx as usize]],
                true,
                u64::MAX,
            )?;
        }

        self.device.wait_for_fences(
            &[self.data.in_flight_fences[self.frame_in_flight_idx]],
            true,
            u64::MAX,
        )?;

        let image_acquisition_result = self
            .device
            .acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX,
                self.data.image_available_semaphores[self.frame_in_flight_idx],
                vk::Fence::null(),
            );

        let image_index = match image_acquisition_result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e))
        };

        self.data.in_flight_swapchain_image_fences[image_index] =
            self.data.in_flight_fences[self.frame_in_flight_idx];

        let wait_semaphores = &[self.data.image_available_semaphores[self.frame_in_flight_idx]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index as usize]];
        let signal_semaphores = &[self.data.render_finished_semaphores[self.frame_in_flight_idx]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device
            .reset_fences(&[self.data.in_flight_fences[self.frame_in_flight_idx]])?;

        self.device.queue_submit(
            self.data.graphics_queue,
            &[submit_info],
            self.data.in_flight_fences[self.frame_in_flight_idx],
        )?;

        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let queue_present_result = self.device
            .queue_present_khr(self.data.present_queue, &present_info);

        let changed = matches!(queue_present_result, Ok(vk::SuccessCode::SUBOPTIMAL_KHR) | Err(vk::ErrorCode::OUT_OF_DATE_KHR));
        if changed || self.resized {
            self.resized = false;
            self.recreate_swapchain(window)?;
        }
        let _queue_present_result = queue_present_result?;

        self.frame_in_flight_idx = (self.frame_in_flight_idx + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }
}

unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let color_attachments = &[color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments);

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE);

    let attachments = &[color_attachment];
    let subpasses = &[subpass];
    let dependencies = &[dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.main_render_pass = unsafe { device.create_render_pass(&info, None)? };

    Ok(())
}

unsafe fn create_framebuffers(device: &Device, data: &mut AppData) -> Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.main_render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            unsafe { device.create_framebuffer(&create_info, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data.surface, data.physical_device)?;

    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::empty()) // Optional.
        .queue_family_index(indices.graphics);

    data.command_pool = device.create_command_pool(&info, None)?;

    Ok(())
}

unsafe fn create_command_buffers(device: &Device, data: &mut AppData) -> Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;

    for (i, cmd) in data.command_buffers.iter().enumerate() {
        let inheritance = vk::CommandBufferInheritanceInfo::builder();

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::empty()) // Optional.
            .inheritance_info(&inheritance); // Optional.

        device.begin_command_buffer(*cmd, &info)?;

        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(data.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let clear_values = &[color_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(data.main_render_pass)
            .framebuffer(data.framebuffers[i])
            .render_area(render_area)
            .clear_values(clear_values);

        device.cmd_begin_render_pass(*cmd, &info, vk::SubpassContents::INLINE);
        device.cmd_bind_pipeline(
            *cmd,
            vk::PipelineBindPoint::GRAPHICS,
            data.render_pipeline.pipeline,
        );
        device.cmd_draw(*cmd, 3, 1, 0, 0);
        device.cmd_end_render_pass(*cmd);
        device.end_command_buffer(*cmd)?;
    }

    Ok(())
}

unsafe fn create_sync_objects(device: &Device, data: &mut AppData) -> Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    data.in_flight_swapchain_image_fences = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

impl Drop for App {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();

            self.data
                .in_flight_fences
                .iter()
                .for_each(|f| self.device.destroy_fence(*f, None));

            self.data
                .render_finished_semaphores
                .iter()
                .for_each(|s| self.device.destroy_semaphore(*s, None));
            self.data
                .image_available_semaphores
                .iter()
                .for_each(|s| self.device.destroy_semaphore(*s, None));

            self.destroy_swapchain_and_pipeline();

            self.device
                .destroy_command_pool(self.data.command_pool, None);

            self.device.destroy_device(None);

            if VALIDATION_ENABLED {
                self.instance
                    .destroy_debug_utils_messenger_ext(self.data.messenger, None);
            }

            self.instance.destroy_surface_khr(self.data.surface, None);
            self.instance.destroy_instance(None);

            info!("Vulkan instance destroyed successfully.");
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AppData {
    pub physical_device: vk::PhysicalDevice,
    pub messenger: vulkanalia_sys::DebugUtilsMessengerEXT,

    pub surface: vk::SurfaceKHR,

    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,

    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain_image_views: Vec<vk::ImageView>,

    pub framebuffers: Vec<vk::Framebuffer>,

    pub main_render_pass: vk::RenderPass,
    pub render_pipeline: AppPipeline,

    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,

    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,

    pub in_flight_fences: Vec<vk::Fence>,
    pub in_flight_swapchain_image_fences: Vec<vk::Fence>,
}
