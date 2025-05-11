use vulkanalia::Version;

pub const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
pub const EXTENSION_VALIDATION_LAYER: vulkanalia_sys::ExtensionName =
    vulkanalia_sys::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
pub const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);