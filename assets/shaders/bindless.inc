#ifndef RENDERTOY_BINDLESS_INC
#define RENDERTOY_BINDLESS_INC

layout(set = 30, binding = 0) uniform texture2D all_textures[];
layout(set = 31, binding = 0) uniform usamplerBuffer all_buffers[];

#endif
