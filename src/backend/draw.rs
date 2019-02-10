use super::shader::{make_program, make_shader};

pub fn draw_fullscreen_texture(tex: u32) {
    lazy_static! {
        static ref PROG: u32 = {
            let vs = make_shader(
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
						Out_Color = textureLod(Texture, Frag_UV * (vec2(texSize - 1) / vec2(texSize)) + vec2(0.5, 0.5) / texSize, 0);
					}"#
                    .to_string(),
                }],
            )
            .expect("Pixel shader failed to compile");

            make_program(&[vs, ps]).expect("Shader failed to link")
        };
    }

    unsafe {
        gl::UseProgram(*PROG);

        gl::ActiveTexture(gl::TEXTURE0);
        gl::BindTexture(gl::TEXTURE_2D, tex);

        let loc = gl::GetUniformLocation(*PROG, "Texture\0".as_ptr() as *const i8);
        let img_unit = 0;
        gl::Uniform1i(loc, img_unit);

        //gl::Viewport(0, 0, width as i32, height as i32);

        gl::DrawArrays(gl::TRIANGLES, 0, 3);
        gl::UseProgram(0);
    }
}
