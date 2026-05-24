/**
 * edgeFlow.js - ONNX Runtime Backend
 *
 * Uses onnxruntime-web for real ONNX model inference.
 * onnxruntime-web is an optional peer dependency loaded dynamically.
 */
import { Runtime, RuntimeType, RuntimeCapabilities, LoadedModel, ModelLoadOptions, Tensor } from '../core/types.js';
/**
 * Override paths for the onnxruntime-web WASM assets, mirroring ORT's
 * `env.wasm.wasmPaths` object form. Set via {@link configureOnnxAssets}
 * when the consumer's bundler emits content-hashed asset URLs (a bare
 * directory prefix can't address those).
 */
export interface OnnxAssetPaths {
    /** URL/path for the main `.wasm` binary. */
    wasm?: string;
    /** URL/path for the loader `.mjs` glue. */
    mjs?: string;
}
/**
 * Inject a pre-imported onnxruntime-web module. Use this in Web Worker
 * contexts where the bundler may not preserve the internal dynamic
 * import. Pass the module from `import * as ort from "onnxruntime-web/wasm"`.
 */
export declare function setOnnxModule(module: any): void;
/**
 * Configure the onnxruntime-web WASM asset locations. Applied to
 * `env.wasm.wasmPaths` during runtime initialization. Necessary when the
 * consumer's bundler content-hashes the `.wasm` / `.mjs` files (e.g. Vite
 * `?url` imports), since the default `/ort/` prefix cannot resolve hashed
 * filenames.
 */
export declare function configureOnnxAssets(paths: OnnxAssetPaths): void;
/**
 * Check whether onnxruntime-web is importable.
 */
export declare function isOnnxAvailable(): Promise<boolean>;
/**
 * ONNXRuntime - Real ONNX model inference using onnxruntime-web
 */
export declare class ONNXRuntime implements Runtime {
    readonly name: RuntimeType;
    private initialized;
    private executionProvider;
    get capabilities(): RuntimeCapabilities;
    /**
     * Check if ONNX Runtime is available (peer dependency installed)
     */
    isAvailable(): Promise<boolean>;
    /**
     * Initialize the ONNX runtime
     */
    initialize(): Promise<void>;
    /**
     * Load a model from ArrayBuffer
     */
    loadModel(modelData: ArrayBuffer, options?: ModelLoadOptions): Promise<LoadedModel>;
    /**
     * Run inference
     */
    run(model: LoadedModel, inputs: Tensor[]): Promise<Tensor[]>;
    /**
     * Run inference with named inputs
     */
    runNamed(model: LoadedModel, namedInputs: Map<string, Tensor>): Promise<Tensor[]>;
    /**
     * Unload a model
     */
    private unloadModel;
    /**
     * Dispose the runtime
     */
    dispose(): void;
}
/**
 * Create ONNX runtime factory
 */
export declare function createONNXRuntime(): Runtime;
//# sourceMappingURL=onnx.d.ts.map