"""Checkpoint prediction and preloaded-RGB resource measurement."""
import json
import math
import statistics
import time
from pathlib import Path
from jnda.controls.io import Images, load_rows, sha256, write_json

SHORT_PROMPT = 'Segment the input agricultural image into background, crop plants, and weeds, preserving pixel alignment.'


def native_forward_at(transformer, packed, *, prompt, text_ids, timestep):
    import torch
    batch = packed.shape[0]
    h = torch.arange(32, dtype=torch.int64)
    w = torch.arange(32, dtype=torch.int64)
    ids = torch.cartesian_prod(torch.arange(1, dtype=torch.int64), h, w,
                              torch.arange(1)).unsqueeze(0).expand(batch, -1, -1).to(packed.device)
    result = transformer(hidden_states=packed, encoder_hidden_states=prompt.expand(batch, -1, -1),
                         timestep=torch.full((batch,), timestep, dtype=torch.float32, device=packed.device),
                         img_ids=ids, txt_ids=text_ids.expand(batch, -1, -1),
                         guidance=torch.zeros(batch, dtype=torch.float32, device=packed.device), return_dict=False)[0]
    if tuple(result.shape) != (batch, 1024, 128):
        raise RuntimeError('Native transformer output shape changed')
    return result


class Predictor:
    def __init__(self, args):
        import numpy as np
        import torch
        from jnda.controls import native, upernet
        from jnda.controls.terminal import NativeOnlyBackbone
        from jnda.controls.runtime import configure, history, model_identity
        self.args, self.np, self.torch = args, np, torch
        self.native, self.upernet = native, upernet
        self.backbone = self.head = self.model = self.legacy = None
        legacy = args.route in ('Fresh_S7', 'Fresh_Arm_B')
        self.backend = configure(args.device, legacy=legacy)
        if args.route != 'B1':
            self.old = history(args.model_root)
            self.old.seed_all(args.seed)
        if legacy:
            if not args.legacy_output:
                raise ValueError('Sensitivity requires the original epoch10 checkpoint directory tree via --legacy-output')
            if args.prompt == 'short':
                self.old.PROMPT = SHORT_PROMPT
            elif args.prompt == 'empty':
                self.old.PROMPT = ''
            self.legacy = self.old._Predictor(root=Path(args.data_root), output=Path(args.legacy_output),
                                              system=args.route, seed=args.seed, s7_checkpoint=None)
            # Pipeline has no callable parameters() in the recorded version.
            for name in ('model', 'transformer', 'vae', 'head'):
                module = getattr(self.legacy, name, None)
                if module is not None:
                    module.eval().requires_grad_(False)
            if args.route == 'Fresh_S7':
                for name in ('transformer', 'vae'):
                    getattr(self.legacy.pipeline, name).eval().requires_grad_(False)
            self.checkpoint_sha256 = sha256(self.legacy.checkpoint)
            return
        if not args.checkpoint:
            raise ValueError('Supplementary prediction requires --checkpoint')
        self.checkpoint_sha256 = sha256(args.checkpoint)
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        cfg = state.get('config', state.get('identity', {}))
        if cfg.get('route') != args.route or cfg.get('seed') != args.seed:
            raise ValueError('Checkpoint route/seed mismatch')
        if args.route == 'B1':
            self.model_identity = model_identity(args.model_root, 'upernet')
            self.model, _ = upernet.build_model(args.model_root, args.seed)
            self.model.load_state_dict(state['model'], strict=True)
            self.model.to(args.device).eval().requires_grad_(False)
        else:
            cls = native.NativeBackbone if args.route == 'T4' else NativeOnlyBackbone
            self.backbone = cls(self.old, Path(args.data_root), lora=args.route == 'T3', seed=args.seed)
            self.head = self.old._native_head(args.seed, device='cpu') if args.route == 'T1' else native.SpatialHead(3072 if args.route == 'T4' else 128)
            self.head.load_state_dict(state['head'], strict=True)
            if args.route == 'T3':
                self.old._load_lora_state(self.backbone.transformer, state['lora'])
            self.head.to(args.device).eval().requires_grad_(False)
            self.backbone.transformer.eval().requires_grad_(False)
            self.backbone.vae.eval().requires_grad_(False)

    def predict(self, rgb, sample_id):
        torch, args = self.torch, self.args
        if self.legacy is not None:
            seed = self.old._sample_seed(args.seed, sample_id)
            if args.route == 'Fresh_S7' and args.steps != 10:
                from PIL import Image
                generator = torch.Generator(device='cuda:0').manual_seed(seed)
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = self.legacy.pipeline(image=Image.fromarray(rgb), prompt_embeds=self.legacy.s7_positive,
                        negative_prompt_embeds=self.legacy.s7_negative, height=512, width=512,
                        num_inference_steps=args.steps, guidance_scale=4.0, generator=generator, output_type='pil').images[0]
                pixels = self.np.asarray(output.convert('RGB'), dtype=self.np.int32)
                distances = pixels[:, :, None, :] - self.old.PALETTE.astype(self.np.int32)[None, None]
                return self.np.square(distances).sum(axis=3).argmin(axis=2).astype(self.np.uint8)
            if args.route == 'Fresh_Arm_B' and args.timestep != 0.999:
                with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    packed = self.old._packed_input(self.legacy.pipeline, self.legacy.vae, rgb)
                    features = native_forward_at(self.legacy.transformer, packed, prompt=self.legacy.prompt,
                                                  text_ids=self.legacy.text_ids, timestep=args.timestep)
                    return self.old._head_logits(self.legacy.head, features).argmax(1).squeeze(0).byte().cpu().numpy()
            return self.legacy.predict(rgb, sample_seed=seed)
        logits, _, _ = self.project(rgb)
        return logits.argmax(1).squeeze(0).byte().cpu().numpy()

    def project(self, rgb):
        torch, args = self.torch, self.args
        if args.route == 'B1':
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                logits = self.model(pixel_values=self.upernet.normalize_rgb(rgb).unsqueeze(0).to(args.device)).logits
        else:
            # The VAE remains outside the transformer/head autocast context.
            with torch.no_grad():
                packed = self.old._packed_input(self.backbone.pipeline, self.backbone.vae, rgb)
            self.backbone.tap = None
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                features = self.old._native_forward(self.backbone.transformer, packed,
                                                   prompt=self.backbone.prompt, text_ids=self.backbone.text_ids)
                tap = self.backbone.tap if args.route == 'T4' else None
                self.backbone.tap = None
                logits = self.old._head_logits(self.head, features) if args.route == 'T1' else self.native.unpatchify(self.head(tap if args.route == 'T4' else features))
        return logits, features if args.route != 'B1' else None, tap if args.route == 'T4' else None


    def parameter_counts(self):
        modules = {}
        if self.legacy is not None:
            if self.args.route == 'Fresh_S7':
                modules = {name: getattr(self.legacy.pipeline, name) for name in ('transformer', 'vae')}
            else:
                modules = {name: getattr(self.legacy, name) for name in ('transformer', 'vae', 'head')}
        elif self.model is not None:
            modules = {'model': self.model}
        else:
            modules = {'transformer': self.backbone.transformer, 'vae': self.backbone.vae, 'head': self.head}
        return {name: sum(p.numel() for p in module.parameters()) for name, module in modules.items()
                if module is not None and callable(getattr(module, 'parameters', None))}

    def inspect(self, rgb, sample_id):
        if self.legacy is not None:
            return self.predict(rgb, sample_id)
        logits, features, tap = self.project(rgb)
        torch, route = self.torch, self.args.route
        # Preserve the original warmup order for the two cached-head controls.
        result = logits.argmax(1).squeeze(0).byte().cpu().numpy() if route in ('T1', 'T4') else None
        if tuple(logits.shape) != (1, 3, 512, 512) or not bool(torch.isfinite(logits).all()):
            raise FloatingPointError('Invalid or nonfinite prediction logits')
        if features is not None and (tuple(features.shape) != (1, 1024, 128) or features.dtype != torch.bfloat16 or not bool(torch.isfinite(features).all())):
            raise FloatingPointError('Invalid terminal feature tensor')
        if route == 'T4' and (tuple(tap.shape) != (1, 1024, 3072) or tap.dtype != torch.float32 or not bool(torch.isfinite(tap).all())):
            raise FloatingPointError('Invalid actual FP32 projection-input feature')
        return result if result is not None else logits.argmax(1).squeeze(0).byte().cpu().numpy()



def run(args):
    from PIL import Image
    import numpy as np
    from jnda.controls.io import load_rows
    rows = load_rows(args.manifest)
    if args.panel_ids:
        ids = Path(args.panel_ids).read_text().splitlines()
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate panel IDs')
        index = {r['sample_id']: r for r in rows}
        rows = [index[s] for s in ids]
    if args.route in ('T1', 'T2', 'T3', 'T4', 'B1') and (args.prompt != 'original' or args.steps != 10 or args.timestep != 0.999):
        raise ValueError('Supplementary controls use fixed conditioning; sensitivity flags apply only to the original sensitivity routes')
    if args.route == 'Fresh_S7' and (args.steps not in (1, 2, 4, 10) or args.prompt != 'original' or args.timestep != 0.999):
        raise ValueError('S7 sensitivity varies only steps1/2/4/10')
    if args.route == 'Fresh_Arm_B' and (args.steps != 10 or args.timestep not in (0.5, 0.75, 0.999) or (args.prompt != 'original' and args.timestep != 0.999)):
        raise ValueError('JNDA sensitivity uses the five predeclared settings, not a cross-product')
    if args.route in ('Fresh_S7', 'Fresh_Arm_B') and args.seed not in (1701, 1702, 1703):
        raise ValueError('Sensitivity has three fixed seeds')
    if args.route in ('Fresh_S7', 'Fresh_Arm_B') and not args.timing:
        if len(rows) != 92 or len({(r['source'], r['session_id']) for r in rows}) != 92:
            raise ValueError('Sensitivity requires the fixed92-image/92-session panel')
    if args.route == 'Fresh_Arm_B' and args.timing:
        raise ValueError('The published sensitivity resource matrix varies S7 steps only')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    data = Images(args.data_root)
    predictor = Predictor(args)
    torch = predictor.torch
    if args.timing:
        if len(rows) != (40 if args.panel_kind == 'main' else 20):
            raise ValueError('Timing requires the frozen40-image main panel or20-image extra panel')
        prepared = [(r, data.rgb(r)) for r in rows]
        for i in range(10):
            r, rgb = prepared[i % len(prepared)]
            value = predictor.inspect(rgb, r['sample_id'])
            if value.shape != (512, 512) or value.dtype != np.uint8 or not np.isin(value, [0, 1, 2]).all():
                raise ValueError('Invalid warmup categorical map')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        records = []
        for row, rgb in prepared:
            for repeat in (1, 2, 3):
                torch.cuda.synchronize()
                tick = time.perf_counter()
                value = predictor.predict(rgb, row['sample_id'])
                torch.cuda.synchronize()
                duration = time.perf_counter() - tick
                if duration <= 0 or not math.isfinite(duration) or value.shape != (512, 512) or value.dtype != np.uint8 or not np.isin(value, [0, 1, 2]).all():
                    raise ValueError('Invalid timed categorical output')
                records.append({'sample_id': row['sample_id'], 'repeat': repeat, 'seconds': duration})
        latency = statistics.median([statistics.median([r['seconds'] for r in records if r['sample_id'] == row['sample_id']]) for row in rows])
        write_json(output / 'resources.json', {'route': args.route, 'seed': args.seed, 'steps': args.steps,
                   'panel_kind': args.panel_kind, 'backend': predictor.backend, 'checkpoint_sha256': predictor.checkpoint_sha256, 'parameters': predictor.parameter_counts(), 'warmups': 10, 'records': records,
                   'panel_latency_seconds': latency, 'serial_throughput': len(records) / sum(r['seconds'] for r in records),
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                   'boundary': 'preloaded resized RGB through model to CPU categorical map; load,prompt,disk,warmups excluded',
                   'GT_access': False, 'predictions_saved': False})
    else:
        index = []
        for row in rows:
            prediction = predictor.inspect(data.rgb(row), row['sample_id'])
            if prediction.shape != (512, 512) or prediction.dtype != np.uint8 or not np.isin(prediction, [0, 1, 2]).all():
                raise ValueError('Invalid categorical prediction')
            name = row['sample_id'] + '.png'
            if Path(name).name != name:
                raise ValueError('Unsafe sample identifier')
            Image.fromarray(prediction).save(output / name)
            index.append({'sample_id': row['sample_id'], 'source': row['source'], 'session_id': row['session_id'],
                          'path': name, 'sha256': sha256(output / name)})
        write_json(output / 'predictions.json', {'system': args.system or args.route, 'seed': args.seed,
                   'checkpoint_sha256': predictor.checkpoint_sha256,
                   'steps': args.steps, 'prompt': args.prompt, 'timestep': args.timestep, 'rows': index})
