use gl::types::*;
use regex::Regex;
use snoozy::Result;
use std::ffi::CString;
use std::iter;

pub(crate) fn make_shader(
    shader_type: u32,
    sources: &[shader_prepper::SourceChunk],
) -> Result<u32> {
    unsafe {
        let handle = gl::CreateShader(shader_type);

        let preamble = shader_prepper::SourceChunk {
            source: "#version 430\n".to_string(),
            file: String::new(),
            line_offset: 0,
        };

        let mut source_lengths: Vec<GLint> = Vec::new();
        let mut source_ptrs: Vec<*const GLchar> = Vec::new();

        let mod_sources: Vec<_> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| shader_prepper::SourceChunk {
                source: format!("#line 0 {}\n", i + 1) + &s.source,
                line_offset: s.line_offset,
                file: s.file.clone(),
            })
            .collect();

        for s in iter::once(&preamble).chain(mod_sources.iter()) {
            source_lengths.push(s.source.len() as GLint);
            source_ptrs.push(s.source.as_ptr() as *const GLchar);
        }

        gl::ShaderSource(
            handle,
            source_ptrs.len() as i32,
            source_ptrs.as_ptr(),
            source_lengths.as_ptr(),
        );
        gl::CompileShader(handle);

        let mut shader_ok: gl::types::GLint = 1;
        gl::GetShaderiv(handle, gl::COMPILE_STATUS, &mut shader_ok);

        if shader_ok != 1 {
            let mut log_len: gl::types::GLint = 0;
            gl::GetShaderiv(handle, gl::INFO_LOG_LENGTH, &mut log_len);

            let log_str = CString::from_vec_unchecked(vec![b'\0'; (log_len + 1) as usize]);

            gl::GetShaderInfoLog(
                handle,
                log_len,
                std::ptr::null_mut(),
                log_str.as_ptr() as *mut gl::types::GLchar,
            );

            let log_str = log_str.to_string_lossy().into_owned();

            lazy_static! {
                static ref intel_error_re: Regex =
                    Regex::new(r"(?m)^ERROR:\s*(\d+):(\d+)").unwrap();
            }

            lazy_static! {
                static ref nv_error_re: Regex = Regex::new(r"(?m)^(\d+)\((\d+)\)\s*").unwrap();
            }

            let error_replacement = |captures: &regex::Captures| -> String {
                let chunk = captures[1].parse::<usize>().unwrap().max(1) - 1;
                let line = captures[2].parse::<usize>().unwrap();
                format!(
                    "{}({})",
                    sources[chunk].file,
                    line + sources[chunk].line_offset
                )
            };

            let pretty_log = intel_error_re.replace_all(&log_str, error_replacement);
            let pretty_log = nv_error_re.replace_all(&pretty_log, error_replacement);

            gl::DeleteShader(handle);
            Err(format_err!(
                "Shader failed to compile: {}",
                pretty_log.to_string()
            ))
        } else {
            Ok(handle)
        }
    }
}

pub(crate) fn make_program(shaders: &[u32]) -> Result<u32> {
    unsafe {
        let handle = gl::CreateProgram();
        for &shader in shaders.iter() {
            gl::AttachShader(handle, shader);
        }

        gl::LinkProgram(handle);

        let mut program_ok: gl::types::GLint = 1;
        gl::GetProgramiv(handle, gl::LINK_STATUS, &mut program_ok);

        if program_ok != 1 {
            let mut log_len: gl::types::GLint = 0;
            gl::GetProgramiv(handle, gl::INFO_LOG_LENGTH, &mut log_len);

            let log_str = CString::from_vec_unchecked(vec![b'\0'; (log_len + 1) as usize]);

            gl::GetProgramInfoLog(
                handle,
                log_len,
                std::ptr::null_mut(),
                log_str.as_ptr() as *mut gl::types::GLchar,
            );

            let log_str = log_str.to_string_lossy().into_owned();

            gl::DeleteProgram(handle);
            Err(format_err!("Shader failed to link: {}", log_str))
        } else {
            Ok(handle)
        }
    }
}
