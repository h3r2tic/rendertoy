Texture2D input_tex : register(t0);
RWTexture2D<float4> output_tex : register(u0);

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
    float4 v = input_tex[DispatchID];
    v.rgb = linear_to_srgb(v.rgb);
    output_tex[DispatchID] = v;
}
