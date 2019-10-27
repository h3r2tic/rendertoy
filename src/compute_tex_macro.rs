#![allow(dead_code)]

use crate::shader::*;
use crate::texture::*;
use snoozy::*;
use std::collections::VecDeque;

#[allow(dead_code)]
#[derive(Debug)]
pub struct ImageFilterUniform {
    pub name: String,
    pub value: ShaderUniformValue,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum ImageFilterToken {
    Uniform(ImageFilterUniform),
    TextureSample(ImageFilterUniform),
    Expr(String),
}

pub fn image_filter_token_join(
    item: ImageFilterToken,
    mut rest: VecDeque<ImageFilterToken>,
) -> VecDeque<ImageFilterToken> {
    rest.push_front(item);
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
                ImageFilterToken::Uniform(ImageFilterUniform { name, .. }) => name.clone(),
                ImageFilterToken::TextureSample(ImageFilterUniform { name, .. }) => {
                    format!("textureLod({}, uv, 0)", name)
                }
                ImageFilterToken::Expr(s) => s.clone(),
            })
            .collect::<Vec<String>>()
            .concat()
    }

    fn declare_uniforms(&self) -> String {
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
                        ShaderUniformValue::Float32Asset(_) => "float",
                        ShaderUniformValue::Uint32Asset(_) => "uint",
                        ShaderUniformValue::UsizeAsset(_) => "int", // TOOO
                        ShaderUniformValue::TextureAsset(_) => "sampler2D",
                        ShaderUniformValue::BufferAsset(_) => {
                            panic!("Buffer parameters not supported")
                        }
                        ShaderUniformValue::Bundle(_) => panic!("Bundle parameters not supported"),
                        ShaderUniformValue::BundleAsset(_) => {
                            panic!("Bundle asset parameters not supported")
                        }
                    };

                    Some(format!("uniform {} {};\n", t, name))
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
uniform restrict writeonly image2D outputTex;
uniform vec4 outputTex_size;

layout (local_size_x = 8, local_size_y = 8) in;
void main() {{
    ivec2 pix = ivec2(gl_GlobalInvocationID.xy);
    vec2 uv = (vec2(pix) + 0.5) * outputTex_size.zw;
    vec4 color = vec4(0, 0, 0, 1);
    {tokens};
    imageStore(outputTex, pix, color);
}}",
            uniforms = self.declare_uniforms(),
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
    (@expr # @ $var:ident $($tts:tt)*) => {
        $crate::compute_tex_macro::image_filter_token_join(
            $crate::compute_tex_macro::ImageFilterToken::TextureSample($crate::compute_tex_macro::ImageFilterUniform {
                name: stringify!($var).to_owned(),
                value: $var.into(),
            }),
            compute_tex!(@expr $($tts)*)
        )
    };
    // An interpolated variable; must be a shader uniform
    (@expr # $var:ident $($tts:tt)*) => {
        $crate::compute_tex_macro::image_filter_token_join(
            $crate::compute_tex_macro::ImageFilterToken::Uniform($crate::compute_tex_macro::ImageFilterUniform {
                name: stringify!($var).to_owned(),
                value: $var.into(),
            }),
            compute_tex!(@expr $($tts)*)
        )
    };
    // Parenthesized expressions. We must descend in order to perform nested interpolations
    (@term ( $($tts:tt)* ) ) => {
        {
            let mut res = compute_tex!(@expr $($tts)*);
            res.push_front($crate::compute_tex_macro::ImageFilterToken::Expr("(".to_owned()));
            res.push_back($crate::compute_tex_macro::ImageFilterToken::Expr(")".to_owned()));
            $crate::compute_tex_macro::ImageFilterTermToken::List(res)
        }
    };
    // A regular item
    (@term $item:tt ) => {
        $crate::compute_tex_macro::ImageFilterTermToken::Single(
            $crate::compute_tex_macro::ImageFilterToken::Expr(stringify!($item).to_owned())
        )
    };
    // Munch a single item. Split off into @term and then process the rest
    (@expr $item:tt $($tts:tt)*) => {
        {
            let term = compute_tex!(@term $item);
            match term {
                $crate::compute_tex_macro::ImageFilterTermToken::Single(expr) => {
                    $crate::compute_tex_macro::image_filter_token_join(
                        expr,
                        compute_tex!(@expr $($tts)*)
                    )
                }
                $crate::compute_tex_macro::ImageFilterTermToken::List(mut list) => {
                    let mut rest = compute_tex!(@expr $($tts)*);
                    list.append(&mut rest);
                    list
                }
            }
        }
    };
    (@swizzle $swizzle:ident) => {
        ".".to_owned() + stringify!($swizzle)
    };
    (@swizzle) => {
        "".to_owned()
    };
    (@munch_bindings $(# $binding:ident : $value:expr,)*) => {
        $(let $binding = $value;)*
    };
    (
        $debug_name:expr,
        $tex_key:expr,
        $(# $binding:ident : $binding_value:expr,)*
        color
        $(. $swizzle:ident)? = $($tts:tt)*
    ) => {
        {
            compute_tex!(@munch_bindings $(# $binding : $binding_value,)*);

            let mut tokens = compute_tex!(@expr $($tts)*);
            let output_str = "color".to_owned() + &compute_tex!(@swizzle $($swizzle)?) + " = ";
            tokens.push_front($crate::compute_tex_macro::ImageFilterToken::Expr(output_str));

            $crate::compute_tex_macro::ImageFilterDesc::new($debug_name.to_owned(), $tex_key, tokens).run()
        }
    };
    () => {
        vec![]
    };
}
