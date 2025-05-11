use anyhow::Result;
use log::trace;
use std::fs;
use std::path::Path;
use vulkanalia::bytecode::Bytecode;
use vulkanalia::vk::{DeviceV1_0, HasBuilder};
use vulkanalia::{Device, vk};
use vulkanalia_sys::{Extent2D, Handle, RenderPass, ShaderModule};

#[derive(Clone, Debug, Default)]
pub struct AppShader {
    pub module: ShaderModule,
}

impl AppShader {
    pub fn new(device: &Device, bytecode: &[u8]) -> Result<Self> {
        let bytecode_aligned = Bytecode::new(bytecode)?;
        // FIXME: why is the code_size parameter required here?
        // we get a validation error otherwise because the code size field is not populated, but the
        // tutorial implies .code() alone is sufficient
        let info = vk::ShaderModuleCreateInfo::builder()
            .code(bytecode_aligned.code())
            .code_size(bytecode_aligned.code_size());

        unsafe {
            Ok(AppShader {
                module: device.create_shader_module(&info, None)?,
            })
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        if !self.module.is_null() {
            unsafe { device.destroy_shader_module(self.module, None) }
        }
    }

    pub fn from_path<P: AsRef<Path>>(device: &Device, path: P) -> Result<Self> {
        let bytecode = fs::read(path)?;
        Self::new(device, &bytecode)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AppPipeline {
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl AppPipeline {
    pub unsafe fn new(
        device: &Device,
        viewport_extent: &Extent2D,
        render_pass: RenderPass,
    ) -> Result<Self> {
        trace!("Compiling app pipeline");
        let mut vert = AppShader::from_path(device, "shaders/vert.spv")?;
        let mut frag = AppShader::from_path(device, "shaders/frag.spv")?;

        let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert.module)
            .name(b"main\0");

        let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag.module)
            .name(b"main\0");

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewport = vk::Viewport::builder()
            .x(0.0)
            .y(0.0)
            .width(viewport_extent.width as f32)
            .height(viewport_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0);

        let scissor = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(*viewport_extent);

        let viewports = &[viewport];
        let scissors = &[scissor];
        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::_1);

        let attachment = vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(vk::ColorComponentFlags::all())
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD);

        let attachments = &[attachment];
        let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let layout_info = vk::PipelineLayoutCreateInfo::builder();
        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None)? };

        let stages = &[vert_stage, frag_stage];
        let info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(stages)
            .vertex_input_state(&vertex_input_state)
            .input_assembly_state(&input_assembly_state)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization_state)
            .multisample_state(&multisample_state)
            .color_blend_state(&color_blend_state)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        let pipeline = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
                .0[0]
        };

        unsafe {
            vert.destroy(device);
            frag.destroy(device);
        }

        Ok(AppPipeline {
            pipeline_layout,
            pipeline,
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
