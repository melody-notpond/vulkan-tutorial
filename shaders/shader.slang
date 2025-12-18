struct UniformBuffer {
  float4x4 model;
  float4x4 view;
  float4x4 proj;
}

struct VsInput {
  float2 pos;
  float3 color;
}

struct VsOutput {
  float4 sv_position: SV_Position;
  float3 color;
};

ConstantBuffer<UniformBuffer> ubo;

[shader("vertex")]
VsOutput vert_main(VsInput input) {
  VsOutput output;
  float4x4 mat = mul(ubo.proj, mul(ubo.view, ubo.model));
  output.sv_position = mul(mat, float4(input.pos, 0.0, 1.0));
  output.color = input.color;
  return output;
}

[shader("fragment")]
float4 frag_main(VsOutput vin) : SV_Target {
  return float4(vin.color, 1.0);
}
