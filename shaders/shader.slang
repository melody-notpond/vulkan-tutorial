struct VsInput {
  float2 pos;
  float3 color;
}

struct VsOutput {
  float4 sv_position: SV_Position;
  float3 color;
};

[shader("vertex")]
VsOutput vert_main(VsInput input) {
  VsOutput output;
  output.sv_position = float4(input.pos, 0.0, 1.0);
  output.color = input.color;
  return output;
}

[shader("fragment")]
float4 frag_main(VsOutput vin) : SV_Target {
  return float4(vin.color, 1.0);
}
