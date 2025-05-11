use crate::gfx::context::AppData;
use crate::gfx::queuefamily::QueueFamilyIndices;
use anyhow::Result;
use vulkanalia::vk::{DeviceV1_0, HasBuilder, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{Device, Instance, vk};
use vulkanalia_sys::{Handle/*, SwapchainKHR*/};
use winit::window::Window;

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        data: &AppData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        unsafe {
            Ok(Self {
                capabilities: instance
                    .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
                formats: instance
                    .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
                present_modes: instance
                    .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
            })
        }
    }
}

/// Matches a surface format we want to use for the swapchain.
/// If we fail to find an exact match, return whichever is available as a fallback.
fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    *formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .unwrap_or_else(|| &formats[0])
}

fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .cloned()
        .find(|m| *m == vk::PresentModeKHR::MAILBOX)
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

pub unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut AppData,
) -> Result<()> {
    unsafe {
        let indices = QueueFamilyIndices::get(instance, data.surface, data.physical_device)?;
        let support = SwapchainSupport::get(instance, data, data.physical_device)?;

        let surface_format = get_swapchain_surface_format(&support.formats);
        let present_mode = get_swapchain_present_mode(&support.present_modes);
        let extent = get_swapchain_extent(window, support.capabilities);

        let mut image_count = support.capabilities.min_image_count + 1;
        if support.capabilities.max_image_count != 0 /* no maximum ? */
            && image_count > support.capabilities.max_image_count
        {
            image_count = support.capabilities.max_image_count;
        }

        let mut queue_family_indices = vec![];
        let image_sharing_mode = if indices.graphics != indices.present {
            queue_family_indices.push(indices.graphics);
            queue_family_indices.push(indices.present);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let info = vk::SwapchainCreateInfoKHR::builder()
            .surface(data.surface)
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .present_mode(present_mode)
            .image_extent(extent)
            .image_array_layers(1) // relevant for stereoscopic video etc.
            .image_sharing_mode(image_sharing_mode)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        data.swapchain = device.create_swapchain_khr(&info, None)?;
        data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
        data.swapchain_format = surface_format.format;
        data.swapchain_extent = extent;

        Ok(())
    }
}

pub unsafe fn create_swapchain_image_views(device: &Device, data: &mut AppData) -> Result<()> {
    unsafe {
        data.swapchain_image_views = data
            .swapchain_images
            .iter()
            .map(|i| {
                let components = vk::ComponentMapping::builder()
                    .r(vk::ComponentSwizzle::IDENTITY)
                    .g(vk::ComponentSwizzle::IDENTITY)
                    .b(vk::ComponentSwizzle::IDENTITY)
                    .a(vk::ComponentSwizzle::IDENTITY);

                let subresource_range = vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);

                let info = vk::ImageViewCreateInfo::builder()
                    .image(*i)
                    .view_type(vk::ImageViewType::_2D)
                    .format(data.swapchain_format)
                    .components(components)
                    .subresource_range(subresource_range);

                device.create_image_view(&info, None)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(())
    }
}
