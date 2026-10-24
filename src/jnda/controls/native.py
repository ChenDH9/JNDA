"""Frozen and jointly adapted spatial readouts."""
import contextlib
import gc
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SEED = 2701

CLASS_WEIGHTS = (0.25, 1.3431242219633113, 1.4751680850839344)

MODEL_REVISION = '8c44a2fbef88fae175da65df054db0f901aa9747'

NATIVE_NAMESPACE = 'S8R_Ex01_cleanroom_joint_native_replication_v1_1'

SPATIAL_HEAD_SPEC = {
    'input_projection': {'kernel': 1, 'channels': 256, 'bias': True},
    'input_norm': {'type': 'GroupNorm', 'groups': 32, 'eps': 1e-5, 'affine': True},
    'activation': {'type': 'GELU', 'approximate': 'none'},
    'residual_blocks': 2,
    'residual_conv': {'kernel': 3, 'stride': 1, 'padding': 1, 'padding_mode': 'zeros', 'bias': False},
    'residual_sequence': 'conv/GN32/GELU/conv/GN32/add_skip/GELU',
    'output_projection': {'kernel': 1, 'channels': 768, 'bias': True},
    'output_rearrangement': 'B,32,32,16,16,3 -> B,3,512,512',
}

def state_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        t = value.detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def seed_cpu(seed):
    random.seed(seed); np.random.seed(seed % (2 ** 32)); torch.manual_seed(seed)

class ResidualSpatialBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(32, 256, eps=1e-5, affine=True)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, 256, eps=1e-5, affine=True)
        self.act = nn.GELU(approximate='none')

    def forward(self, value):
        x = self.act(self.norm1(self.conv1(value)))
        return self.act(value + self.norm2(self.conv2(x)))

class SpatialHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.projection = nn.Conv2d(input_dim, 256, 1, bias=True)
        self.norm = nn.GroupNorm(32, 256, eps=1e-5, affine=True)
        self.act = nn.GELU(approximate='none')
        self.blocks = nn.Sequential(ResidualSpatialBlock(), ResidualSpatialBlock())
        self.output = nn.Conv2d(256, 768, 1, bias=True)

    def forward(self, features):
        if tuple(features.shape[1:]) != (1024, self.input_dim):
            raise ValueError(f'feature shape differs: {features.shape}')
        x = features.transpose(1, 2).reshape(-1, self.input_dim, 32, 32)
        x = self.act(self.norm(self.projection(x)))
        return self.output(self.blocks(x)).flatten(2).transpose(1, 2)

def unpatchify(projected):
    if tuple(projected.shape[1:]) != (1024, 768):
        raise ValueError(f'projected shape differs: {projected.shape}')
    return projected.reshape(-1, 32, 32, 16, 16, 3).permute(0, 1, 3, 2, 4, 5).reshape(-1, 512, 512, 3).permute(0, 3, 1, 2).contiguous()

def optimizer_for(head, route, transformer=None, head_lr=1e-3):
    groups = []
    names = []
    for name, p in head.named_parameters():
        # Bias exception is applied to every module, not just the old linear.
        wd = 0.0 if name.endswith('bias') else 0.01
        groups.append({'params': [p], 'lr': head_lr, 'weight_decay': wd})
        names.append({'name': 'head.' + name, 'lr': head_lr, 'weight_decay': wd, 'shape': list(p.shape), 'numel': p.numel(), 'master_dtype':str(p.dtype)})
    if route == 'T3':
        for name, p in transformer.named_parameters():
            if p.requires_grad:
                if 'lora_' not in name.lower():
                    raise RuntimeError('transformer trainable outside LoRA')
                groups.append({'params': [p], 'lr': 1e-4, 'weight_decay': 0.01})
                names.append({'name': 'transformer.' + name, 'lr': 1e-4, 'weight_decay': 0.01, 'shape': list(p.shape), 'numel': p.numel(), 'master_dtype':str(p.dtype)})
    optimizer = torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8, foreach=None)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    return optimizer, scheduler, names

class NativeBackbone:
    def __init__(self, old, old_root, *, lora=False, seed=SEED, device='cuda:0', attention_backend='native'):
        from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline
        self.old = old; self.device = device; self.lora = lora
        old.seed_all(seed)
        self.prompt, self.text_ids = old._prompt_embeddings(Path(old_root), device=device)
        self.pipeline = Flux2KleinPipeline.from_pretrained(str(old._model_root(Path(old_root))), vae=None, text_encoder=None, tokenizer=None, scheduler=None, torch_dtype=torch.bfloat16, local_files_only=True)
        self.transformer = self.pipeline.transformer
        self.transformer.requires_grad_(False)
        if lora:
            from peft import LoraConfig
            from researchos.flux2_klein_i2i.real_trainer import lora_target_modules
            self.transformer.add_adapter(LoraConfig(r=8, lora_alpha=8, lora_dropout=0.0, init_lora_weights='gaussian', target_modules=lora_target_modules(int(self.transformer.config.num_single_layers))))
            for name, p in self.transformer.named_parameters():
                p.requires_grad_('lora_' in name.lower())
            self.transformer.enable_gradient_checkpointing()
        elif getattr(self.transformer, 'peft_config', {}) or any('lora_' in n.lower() for n, _ in self.transformer.named_parameters()):
            raise RuntimeError('pure base route has an adapter')
        self.transformer.set_attention_backend(attention_backend)
        self.pipeline.to(device)
        if lora and any(p.dtype!=torch.bfloat16 for p in self.transformer.parameters() if p.requires_grad):
            raise RuntimeError('actual LoRA master dtype differs from pinned BF16 path')
        self.vae = AutoencoderKLFlux2.from_pretrained(str(old._model_root(Path(old_root))), subfolder='vae', torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval().requires_grad_(False)
        self.transformer.train(lora)
        self.tap = None
        self.tap_hook = self.transformer.proj_out.register_forward_pre_hook(self._tap)
        self.metadata = self.describe()

    def _tap(self, module, args):
        value = args[0]
        if tuple(value.shape[1:]) != (1024, 3072):
            raise RuntimeError(f'proj_out input tap differs: {value.shape}')
        self.tap = value

    def describe(self):
        tr = self.transformer
        return {'adapters': sorted(getattr(tr, 'peft_config', {}).keys()), 'trainable_parameters': [{'name': n, 'shape': list(p.shape), 'numel': p.numel(), 'master_dtype':str(p.dtype)} for n, p in tr.named_parameters() if p.requires_grad], 'transformer_parameters': sum(p.numel() for p in tr.parameters()), 'gradient_checkpointing': bool(tr.is_gradient_checkpointing), 'tap': {'module': 'transformer.proj_out', 'hook': 'forward_pre_hook input[0]', 'after_norm_out': True, 'text_tokens_removed_before_hook': True, 'feature_dim': 3072, 'tokens': 1024, 'order': '32x32 h-major/w-minor from original img_ids'}, 'attention_processors': [{'module': n, 'processor': type(m.processor).__name__, 'backend': str(getattr(m.processor, '_attention_backend', None))} for n, m in tr.named_modules() if hasattr(m, 'processor')], 'prompt_state_sha256': state_hash({'prompt': self.prompt, 'text_ids': self.text_ids}), 'prompt_sha256': hashlib.sha256(self.old.PROMPT.encode()).hexdigest()}

    def features(self, rgb, *, gradients=False):
        # no_grad (rather than inference_mode) leaves features usable by autograd.
        with torch.no_grad():
            packed = self.old._packed_input(self.pipeline, self.vae, rgb)
        self.tap = None
        with contextlib.nullcontext() if gradients else torch.no_grad():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                native = self.old._native_forward(self.transformer, packed, prompt=self.prompt, text_ids=self.text_ids)
        tap = self.tap
        self.tap = None
        if tap is None or tap.shape[-1] != 3072:
            raise RuntimeError('T4 proj_out input was not captured')
        return native, tap

    def close(self):
        self.tap_hook.remove()
        self.pipeline.to('cpu'); self.vae.to('cpu')
        self.prompt = self.text_ids = self.tap = None
        gc.collect(); torch.cuda.empty_cache()

def weighted_loss(logits, labels, device='cuda:0'):
    weights = torch.tensor(CLASS_WEIGHTS, device=device, dtype=torch.float32)
    if not bool((labels != 255).any()):
        raise RuntimeError('all-ignore train item: protocol requires report, no silent skipped update')
    loss = F.cross_entropy(logits.float(), labels, weight=weights, ignore_index=255)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('nonfinite weighted CE')
    return loss

def frozen_fingerprint(model):
    return state_hash({n: p for n, p in model.named_parameters() if not p.requires_grad})
