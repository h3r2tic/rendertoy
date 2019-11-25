Texture2D input_tex : register(t0);
RWTexture2D<float4> output_tex : register(u0);

[numthreads(8, 8, 1)]
void main(in uint2 DispatchID : SV_DispatchThreadID) {
    output_tex[DispatchID] = input_tex[DispatchID];
}
