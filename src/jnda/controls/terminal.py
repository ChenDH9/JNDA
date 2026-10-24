"""Terminal transformer features without retaining the projection-input tap."""
import contextlib
import torch
from . import native

class NativeOnlyBackbone(native.NativeBackbone):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tap_hook.remove()
        self.metadata.update(native_only=True,tap=None,tap4_generated_or_stored=False)
    def features(self,rgb,*,gradients=False):
        with torch.no_grad():
            packed=self.old._packed_input(self.pipeline,self.vae,rgb)
        with contextlib.nullcontext() if gradients else torch.no_grad():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                value=self.old._native_forward(self.transformer,packed,prompt=self.prompt,text_ids=self.text_ids)
        if value.dtype!=torch.bfloat16 or tuple(value.shape)!=(1,1024,128):
            raise RuntimeError('native feature dtype/shape differs')
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('native features nonfinite')
        return value,None
