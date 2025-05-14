use anyhow::Result;
use gfx::context::App;
use winit::event_loop::{ControlFlow, EventLoop};

mod gfx;

// TODO ideas for improvements:
// - refactoring things out of `context`?
// - figure out some form of RAII abstractions?
// - document and research the uses of unsafe{} further?

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::default();
    event_loop.run_app(&mut app)?;

    Ok(())
}
