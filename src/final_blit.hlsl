[[vk::binding(0)]] Texture2D input_tex;
[[vk::binding(1)]] RWTexture2D<float4> output_tex;
[[vk::binding(2)]] SamplerState linear_sampler;

[[vk::push_constant]]
struct {
    float2 input_tex_size;
} push_constants;

float linear_to_srgb(float v) {
    if (v <= 0.0031308) {
        return v * 12.92;
    } else {
        return pow(v, (1.0/2.4)) * (1.055) - 0.055;
    }
}

float3 linear_to_srgb(float3 v) {
	return float3(
		linear_to_srgb(v.x), 
		linear_to_srgb(v.y), 
		linear_to_srgb(v.z));
}

[numthreads(8, 8, 1)]
void main(in uint2 DispatchID : SV_DispatchThreadID) {
    float4 v = input_tex.SampleLevel(
        linear_sampler,
        float2(DispatchID + 0.5) * push_constants.input_tex_size,
        0);
    v.rgb = linear_to_srgb(v.rgb);
    output_tex[DispatchID] = v;
}
