use std::os::raw::c_void;
use std::ffi::CStr;
use log::{debug, error, trace, warn};
use vulkanalia_sys as vksys;

pub extern "system" fn debug_callback(
    severity: vksys::DebugUtilsMessageSeverityFlagsEXT,
    type_: vksys::DebugUtilsMessageTypeFlagsEXT,
    data: *const vksys::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vksys::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vksys::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({type_:?}) {message}");
    } else if severity >= vksys::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({type_:?}) {message}");
    } else if severity >= vksys::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({type_:?}) {message}");
    } else {
        trace!("({type_:?}) {message}");
    }

    vksys::FALSE
}