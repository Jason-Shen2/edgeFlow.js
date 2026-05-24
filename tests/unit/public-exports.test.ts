/**
 * Guards the public package entry point (src/index.ts).
 *
 * Several runtime/ONNX helpers are implemented in submodules but were not
 * re-exported from the barrel, so consumers couldn't import them (see
 * docs/INTEGRATION_LOG.md 2026-05-24). These assertions import from the
 * package entry — not the submodule — so a dropped re-export fails here.
 */
import { describe, it, expect } from 'vitest';
import * as edgeflow from '../../src/index';

describe('public exports (barrel)', () => {
  it('re-exports named-input inference', () => {
    expect(typeof edgeflow.runInferenceNamed).toBe('function');
  });

  it('re-exports the ONNX module/asset injection API', () => {
    expect(typeof edgeflow.setOnnxModule).toBe('function');
    expect(typeof edgeflow.configureOnnxAssets).toBe('function');
  });

  it('setOnnxModule / configureOnnxAssets are callable without throwing', () => {
    expect(() => edgeflow.configureOnnxAssets({ wasm: '/x.wasm', mjs: '/x.mjs' })).not.toThrow();
    expect(() => edgeflow.setOnnxModule({})).not.toThrow();
    // reset injected module so other suites aren't affected
    expect(() => edgeflow.setOnnxModule(null)).not.toThrow();
  });
});
