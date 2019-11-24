use super::shader::{make_program, make_shader};

pub fn draw_fullscreen_texture(gfx: &crate::Gfx, tex: u32, framebuffer_size: (u32, u32)) {
    /*use std::sync::Mutex;

    lazy_static! {
        static ref PROG: Mutex<Option<u32>> = { Mutex::new(None) };
    }

    if PROG.lock().unwrap().is_none() {
        *PROG.lock().unwrap() = {
            let vs = make_shader(
                gl,
                gl::VERTEX_SHADER,
                &[shader_prepper::SourceChunk {
                    file: "no_file".to_string(),
                    line_offset: 0,
                    source: r#"
                        out vec2 Frag_UV;
                        void main()
                        {
                            Frag_UV = vec2(gl_VertexID & 1, gl_VertexID >> 1) * 2.0;
                            gl_Position = vec4(Frag_UV * 2.0 - 1.0, 0, 1);
                        }"#
                    .to_string(),
                }],
            )
            .expect("Vertex shader failed to compile");

            let ps = make_shader(
                gl,
                gl::FRAGMENT_SHADER,
                &[shader_prepper::SourceChunk {
                    file: "no_file".to_string(),
                    line_offset: 0,
                    source: r#"
                        uniform sampler2D Texture;
                        in vec2 Frag_UV;
                        out vec4 Out_Color;
                        void main()
                        {
                            ivec2 texSize = textureSize(Texture, 0);
                            Out_Color = textureLod(Texture, Frag_UV, 0);
                        }"#
                    .to_string(),
                }],
            )
            .expect("Pixel shader failed to compile");

            Some(make_program(gfx, &[vs, ps]).expect("Shader failed to link"))
        };
    }
    let prog = PROG.lock().unwrap().unwrap();

    lazy_static! {
        static ref SAMPLER_ID: Mutex<Option<u32>> = { Mutex::new(None) };
    }

    if SAMPLER_ID.lock().unwrap().is_none() {
        *SAMPLER_ID.lock().unwrap() = unsafe {
            let mut sampler_id = 0;
            gl.GenSamplers(1, &mut sampler_id);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_MAG_FILTER, gl::LINEAR as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_WRAP_S, gl::CLAMP_TO_EDGE as i32);
            gl.SamplerParameteri(sampler_id, gl::TEXTURE_WRAP_T, gl::CLAMP_TO_EDGE as i32);
            Some(sampler_id)
        };
    };
    let sampler_id = SAMPLER_ID.lock().unwrap().unwrap();

    unsafe {
        gl.UseProgram(prog);

        gl.ActiveTexture(gl::TEXTURE0);
        gl.BindSampler(0, sampler_id);
        gl.BindTexture(gl::TEXTURE_2D, tex);

        let loc = gl.GetUniformLocation(prog, "Texture\0".as_ptr() as *const i8);
        let img_unit = 0;
        gl.Uniform1i(loc, img_unit);

        gl.Viewport(0, 0, framebuffer_size.0 as i32, framebuffer_size.1 as i32);
        gl.Disable(gl::DEPTH_TEST);

        gl.DrawArrays(gl::TRIANGLES, 0, 3);
        gl.UseProgram(0);
    }*/
    unimplemented!()
}
