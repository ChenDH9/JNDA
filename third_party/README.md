# Frozen Diffusers distribution

The fixed wheel is a separate dependency asset in the versioned release. Install it instead of substituting a newer Diffusers release:

```bash
python -m pip install --no-deps 'https://github.com/ChenDH9/JNDA/releases/download/v1.0.0/diffusers-0.40.0.dev0-py3-none-any.whl#sha256=e4ba4b112659d61d8254cc03e08caf29e7d59962310b65000d145d6c59ae1d81'
```

SHA256: `e4ba4b112659d61d8254cc03e08caf29e7d59962310b65000d145d6c59ae1d81`

The package contains the exact 996 installed library source files used for the study, totalling 24,858,585 bytes. Packaging metadata provides a relative-path `RUNTIME_SOURCE_MANIFEST.json`, and the wheel's `RECORD` can be verified independently. Runtime source and upstream license texts are unchanged. Descriptive packaging metadata is adapted for this distribution. The package remains under Apache License 2.0, not the JNDA MIT license.

The pinned Linux environment is listed in `environments/requirements-linux-cu128.txt`. This snapshot is provided to preserve the pipeline interfaces and execution behavior used by the experiments. Its version is `0.40.0.dev0`; a matching version label alone does not identify the same source snapshot.

## PhenoBench training example

`diffusers_examples/train_dreambooth_lora_flux2_klein_img2img.py` is the exact frozen training example used by the PhenoBench RGB-LoRA bridge. Its SHA256 is `6486a9d7b56264a047339c3e8291d0d55cae51266bb2c3c5d8d533872330e31e`. The original Apache License 2.0 text is retained in that directory. The source checkout did not retain an upstream Git revision; the supplied source bytes and hash identify the training example used here.
