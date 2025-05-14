use anyhow::Result;
use gfx::context::App;
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::WindowBuilder;

mod gfx;

// TODO ideas for improvements:
// - refactoring things out of `context`?
// - figure out some form of RAII abstractions?
// - document and research the uses of unsafe{} further?

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("gfx :3")
        .with_inner_size(LogicalSize::new(1280, 720))
        .build(&event_loop)?;

    let mut app = unsafe { App::create(&window)? };
    let mut old_size = window.inner_size();
    event_loop.run(move |event, elwt| match event {
        Event::AboutToWait => window.request_redraw(),
        Event::WindowEvent { event, .. } => match event {
            WindowEvent::RedrawRequested if !elwt.exiting() && !app.minimized => {
                unsafe { app.render(&window) }.unwrap()
            }
            WindowEvent::CloseRequested => {
                elwt.exit();
            }
            WindowEvent::Resized(new_size) => {
                if new_size.width == 0 || new_size.height == 0 {
                    app.minimized = true;    
                }
                else if new_size != old_size {
                    app.resized = true;
                    app.minimized = false;
                }
                old_size = new_size;
            }
            _ => {}
        },
        _ => {}
    })?;

    Ok(())
}
