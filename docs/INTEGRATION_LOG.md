# edgeFlow.js Integration Log

A running record of friction hit by downstream consumers integrating
edgeFlow.js — missing APIs, workarounds, confusing errors, bundler
quirks, tokenizer / wasm-path issues. The point is to drive concrete
library improvements instead of letting workarounds rot silently in the
consumer.

The first consumer is **crest** (a terminal app; its NLD module runs a
shell-vs-natural-language classifier in a Web Worker via this library).

## How to use this log

When you hit friction integrating or using edgeFlow.js, add an entry —
**the moment you write the workaround, don't batch.** On the consumer
side, mark the workaround in code with:

```
// TODO(edgeflow): docs/INTEGRATION_LOG.md YYYY-MM-DD — <one-line>
```

so it shows up in `grep` and can be removed once the library catches up.

### Entry template

```
### YYYY-MM-DD — <short title>

**Consumer:** <repo> (<where in the consumer>)
**What I was doing:** <task>
**Friction:** <the missing API / bug / quirk>
**Workaround:** <what the consumer did to ship>
**Upstream fix:** <none yet | commit/PR | shipped in vX.Y.Z>
```

Newest entries first.

---

### 2026-05-24 — ONNX backend unusable from Web Workers (3 related gaps)

**Consumer:** crest (`frontend/app/term/nld/embedder.worker.ts` — the NLD
classifier runs ORT inference in a dedicated Web Worker).

**What I was doing:** running `loadModel()` + named-input inference on a
sequence-classification ONNX model inside a Web Worker, with the WASM
assets served from Vite (content-hashed `?url` imports).

**Friction:** three gaps, all surfacing as the worker silently failing
with "no runtime available" or being unable to address the WASM files:

1. **No way to inject the ORT module.** The backend's
   `getOrt()` does `await import('onnxruntime-web/wasm')`. Vite's worker
   chunker doesn't always preserve that dynamic import inside a worker
   chunk; when it's dropped, `getOrt()` returns null and the backend
   reports "no runtime available." The consumer had a statically-imported
   ORT module in the worker but no way to hand it to edgeFlow.js.

2. **No way to configure WASM asset paths.** The backend hardcoded
   `env.wasm.wasmPaths = '/ort/'`, but Vite emits content-hashed asset
   URLs (e.g. `/assets/ort-wasm-simd-threaded-CDsxkEtH.wasm`) that a bare
   directory prefix can't address. There was no public API to set the
   `{ wasm, mjs }` object form.

3. **WASM config gated on `typeof window !== "undefined"`.** Even when
   ORT loaded, `initialize()` only configured `env.wasm` under a `window`
   check — always false in a Web Worker — so workers got an unconfigured
   runtime. This is the exact context the consumer runs in.

   Bonus: `runInferenceNamed` (named-input inference) is implemented in
   `core/runtime.ts` but was never re-exported from the public entry
   point, so the consumer couldn't import it and fell back to positional
   `runInference` with manually-ordered tensors.

**Workaround (crest):** set `ort.env.wasm.wasmPaths`/`numThreads`
directly on the statically-imported ORT module (relying on ESM singleton
identity so the library's dynamic import inherits it), and use positional
`runInference` ordered against `model.metadata.inputs`.

**Upstream fix (this commit):**
- Added `setOnnxModule(module)` — inject a pre-imported ORT module;
  `getOrt()` prefers it over the dynamic import.
- Added `configureOnnxAssets({ wasm, mjs })` — sets `env.wasm.wasmPaths`
  (object form) during init.
- Removed the `typeof window` gate in `ONNXRuntime.initialize()` so
  worker contexts get configured too (numThreads=1, asset paths).
- Re-exported `runInferenceNamed` from the package entry point.

All four are now public from `edgeflowjs`. Once a release ships, crest
can drop the workaround (search its tree for `TODO(edgeflow)`).
