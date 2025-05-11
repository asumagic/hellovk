use vulkanalia_sys::Handle;
use anyhow::{anyhow, Result};
use vulkanalia::{vk, Instance};
use vulkanalia::vk::{InstanceV1_0, KhrSurfaceExtension};

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32,
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        unsafe {
            let properties = instance.get_physical_device_queue_family_properties(physical_device);

            let graphics = properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32)
                .ok_or_else(|| anyhow!("Device does not have a graphics queue family"))?;

            let mut present = None;
            for (index, _properties) in properties.iter().enumerate() {
                assert!(!surface.is_null());
                if instance.get_physical_device_surface_support_khr(
                    physical_device,
                    index as u32,
                    surface,
                )? {
                    present = Some(index as u32);
                    break;
                }
            }
            let present =
                present.ok_or_else(|| anyhow!("Device does not have a present queue family"))?;

            Ok(Self { graphics, present })
        }
    }
}