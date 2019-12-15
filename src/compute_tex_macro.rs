#![allow(dead_code)]

use crate::shader::*;
use crate::texture::*;
use snoozy::*;
use std::collections::VecDeque;

#[allow(dead_code)]
#[derive(Debug)]
pub struct ImageFilterUniform {
    pub name: &'static str,
    pub value: ShaderUniformValue,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum ImageFilterToken {
    Uniform(ImageFilterUniform),
    TextureSample(ImageFilterUniform),
    Expr(&'static str),
}

pub fn token_join(
    item: ImageFilterToken,
    mut rest: VecDeque<ImageFilterToken>,
) -> VecDeque<ImageFilterToken> {
    rest.push_front(item);
    rest
}

pub fn term_token_join(
    term: ImageFilterTermToken,
    mut rest: VecDeque<ImageFilterToken>,
) -> VecDeque<ImageFilterToken> {
    match term {
        ImageFilterTermToken::Single(expr) => token_join(expr, rest),
        ImageFilterTermToken::List(mut list) => {
            list.append(&mut rest);
            list
        }
    }
}

pub fn parenthesized(mut tokens: VecDeque<ImageFilterToken>) -> ImageFilterTermToken {
    tokens.push_front(ImageFilterToken::Expr("("));
    tokens.push_back(ImageFilterToken::Expr(")"));
    ImageFilterTermToken::List(tokens)
}

pub fn texture_sample(
    name: &'static str,
    value: ShaderUniformValue,
    mut rest: VecDeque<ImageFilterToken>,
) -> VecDeque<ImageFilterToken> {
    rest.push_front(ImageFilterToken::TextureSample(ImageFilterUniform {
        name,
        value,
    }));
    rest
}

pub fn uniform(
    name: &'static str,
    value: ShaderUniformValue,
    mut rest: VecDeque<ImageFilterToken>,
) -> VecDeque<ImageFilterToken> {
    rest.push_front(ImageFilterToken::Uniform(ImageFilterUniform {
        name,
        value,
    }));
    rest
}

#[derive(Debug)]
pub struct ImageFilterDesc {
    pub debug_name: String,
    pub tex_key: TextureKey,
    pub tokens: VecDeque<ImageFilterToken>,
}

impl ImageFilterDesc {
    pub fn new(
        debug_name: String,
        tex_key: TextureKey,
        tokens: VecDeque<ImageFilterToken>,
    ) -> Self {
        Self {
            debug_name,
            tex_key,
            tokens,
        }
    }

    pub fn stringify_tokens(tokens: &VecDeque<ImageFilterToken>) -> String {
        tokens
            .iter()
            .map(|tok| match tok {
                ImageFilterToken::Uniform(ImageFilterUniform { name, .. }) => String::from(*name),
                ImageFilterToken::TextureSample(ImageFilterUniform { name, .. }) => {
                    format!("texelFetch({}, pix, 0)", name)
                }
                ImageFilterToken::Expr(s) => String::from(*s),
            })
            .collect::<Vec<String>>()
            .concat()
    }

    fn declare_cb(&self) -> String {
        self.tokens
            .iter()
            .filter_map(|tok| match tok {
                ImageFilterToken::Uniform(ImageFilterUniform { name, value })
                | ImageFilterToken::TextureSample(ImageFilterUniform { name, value }) => {
                    let t = match value {
                        ShaderUniformValue::Float32(_) => "float",
                        ShaderUniformValue::Uint32(_) => "uint",
                        ShaderUniformValue::Int32(_) => "int",
                        ShaderUniformValue::Ivec2(_) => "ivec2",
                        ShaderUniformValue::Vec4(_) => "vec4",
                        ShaderUniformValue::Float32Asset(_) => "float",
                        ShaderUniformValue::Uint32Asset(_) => "uint",
                        ShaderUniformValue::UsizeAsset(_) => "int", // TOOO
                        ShaderUniformValue::TextureAsset(_) => return None,
                        ShaderUniformValue::BufferAsset(_) => {
                            panic!("Buffer parameters not supported")
                        }
                        ShaderUniformValue::Bundle(_) => panic!("Bundle parameters not supported"),
                        ShaderUniformValue::BundleAsset(_) => {
                            panic!("Bundle asset parameters not supported")
                        }
                    };

                    Some(format!("{} {};\n", t, name))
                }
                _ => None,
            })
            .collect::<Vec<String>>()
            .concat()
    }

    fn declare_uniforms(&self) -> String {
        let mut binding = 1; // will start from 2 due to pre-increment before output
        self.tokens
            .iter()
            .filter_map(|tok| match tok {
                ImageFilterToken::Uniform(ImageFilterUniform { name, value })
                | ImageFilterToken::TextureSample(ImageFilterUniform { name, value }) => {
                    let t = match value {
                        ShaderUniformValue::Float32(_) => return None,
                        ShaderUniformValue::Uint32(_) => return None,
                        ShaderUniformValue::Int32(_) => return None,
                        ShaderUniformValue::Ivec2(_) => return None,
                        ShaderUniformValue::Vec4(_) => return None,
                        ShaderUniformValue::Float32Asset(_) => return None,
                        ShaderUniformValue::Uint32Asset(_) => return None,
                        ShaderUniformValue::UsizeAsset(_) => return None,
                        ShaderUniformValue::TextureAsset(_) => "texture2D",
                        ShaderUniformValue::BufferAsset(_) => {
                            panic!("Buffer parameters not supported")
                        }
                        ShaderUniformValue::Bundle(_) => panic!("Bundle parameters not supported"),
                        ShaderUniformValue::BundleAsset(_) => {
                            panic!("Bundle asset parameters not supported")
                        }
                    };

                    binding += 1;
                    Some(format!(
                        "layout(binding = {}) uniform {} {};\n",
                        binding, t, name
                    ))
                }
                _ => None,
            })
            .collect::<Vec<String>>()
            .concat()
    }

    pub fn run(self) -> SnoozyRef<Texture> {
        let tex_key = self.tex_key;
        let glsl = self.to_glsl();
        let debug_name = self.debug_name;

        let uniforms = self
            .tokens
            .into_iter()
            .filter_map(|tok| match tok {
                ImageFilterToken::Uniform(ImageFilterUniform { name, value })
                | ImageFilterToken::TextureSample(ImageFilterUniform { name, value }) => {
                    Some(ShaderUniformHolder::from_name_value(&name, value))
                }
                _ => None,
            })
            .collect();

        compute_tex(tex_key, load_cs_from_string(glsl, debug_name), uniforms)
    }

    fn to_glsl(&self) -> String {
        format!(
            "{uniforms}
uniform restrict writeonly layout(binding = 0) image2D outputTex;
layout(std140, binding = 1) uniform globals {{
    uniform vec4 outputTex_size;
    {cb}
}};

layout (local_size_x = 8, local_size_y = 8) in;
void main() {{
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pix) + 0.5) * outputTex_size.zw;
    vec4 _output_color = vec4(0, 0, 0, 1);
    {tokens};
    imageStore(outputTex, pix, _output_color);
}}",
            uniforms = self.declare_uniforms(),
            cb = self.declare_cb(),
            tokens = Self::stringify_tokens(&self.tokens)
        )
    }
}

#[allow(dead_code)]
pub enum ImageFilterTermToken {
    Single(ImageFilterToken),
    List(VecDeque<ImageFilterToken>),
}

#[macro_export]
macro_rules! compute_tex {
    (@expr) => {
        std::collections::VecDeque::new()
    };
    // A texture sample from an interpolated variable; must be a shader uniform
    (@expr @ $var:ident $($tts:tt)*) => {
        ctm::texture_sample(
            stringify!($var),
            $var.clone().into(),
            compute_tex!(@expr $($tts)*)
        )
    };
    // An interpolated variable; must be a shader uniform
    (@expr # $var:ident $($tts:tt)*) => {
        ctm::uniform(
            stringify!($var),
            $var.clone().into(),
            compute_tex!(@expr $($tts)*)
        )
    };
    // Parenthesized expressions. We must descend in order to perform nested interpolations
    (@term ( $($tts:tt)* ) ) => {
        ctm::parenthesized(compute_tex!(@expr $($tts)*))
    };
    // A regular item
    (@term $item:tt ) => {
        ctm::ImageFilterTermToken::Single(
            ctm::ImageFilterToken::Expr(stringify!($item))
        )
    };
    // Munch a single item. Split off into @term and then process the rest
    (@expr $item:tt $($tts:tt)*) => {
        {
            let item = compute_tex!(@term $item);
            let rest = compute_tex!(@expr $($tts)*);
            ctm::term_token_join(item, rest)
        }
    };
    (@swizzle $swizzle:ident) => {
        concat!(".", stringify!($swizzle))
    };
    (@swizzle) => {
        ""
    };
    (@munch_bindings $(# $binding:ident : $value:expr,)*) => {
        $(let $binding = $value.clone();)*
    };
    (
        $debug_name:expr,
        $tex_key:expr,
        $(# $binding:ident : $binding_value:expr,)*
        $(. $swizzle:ident)? = $($tts:tt)*
    ) => {
        {
            use $crate::compute_tex_macro as ctm;
            compute_tex!(@munch_bindings $(# $binding : $binding_value,)*);

            let mut tokens = compute_tex!(@expr $($tts)*);
            let output_str = concat!("_output_color", compute_tex!(@swizzle $($swizzle)?), " = ");
            tokens.push_front(ctm::ImageFilterToken::Expr(output_str));

            ctm::ImageFilterDesc::new($debug_name.to_owned(), $tex_key, tokens).run()
        }
    };
    () => {
        vec![]
    };
}
