"""Transformer LoRA target names used by JNDA."""

def lora_target_modules(num_single_layers: int) -> list[str]:
    if num_single_layers <= 0:
        raise ValueError("transformer must expose single transformer layers")
    return ["to_k", "to_q", "to_v", "to_out.0", "to_qkv_mlp_proj"] + [
        f"single_transformer_blocks.{index}.attn.to_out"
        for index in range(num_single_layers)
    ]
