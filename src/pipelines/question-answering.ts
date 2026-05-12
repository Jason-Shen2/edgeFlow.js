/**
 * edgeFlow.js - Question Answering Pipeline
 * 
 * Extract answers from context given a question using real ONNX QA models.
 */

import { BasePipeline, PipelineResult, registerPipeline } from './base.js';
import { EdgeFlowTensor, softmax } from '../core/tensor.js';
import { PipelineConfig, PipelineOptions, LoadedModel } from '../core/types.js';
import { Tokenizer } from '../utils/tokenizer.js';
import { loadModelData } from '../utils/model-loader.js';
import { loadModelFromBuffer, runInferenceNamed } from '../core/runtime.js';

// ============================================================================
// Default Model (DistilBERT fine-tuned on SQuAD)
// ============================================================================

const DEFAULT_MODELS = {
  model: 'https://huggingface.co/Xenova/distilbert-base-cased-distilled-squad/resolve/main/onnx/model_quantized.onnx',
  tokenizer: 'https://huggingface.co/Xenova/distilbert-base-cased-distilled-squad/resolve/main/tokenizer.json',
};

// ============================================================================
// Types
// ============================================================================

export interface QAInput {
  question: string;
  context: string;
}

export interface QuestionAnsweringOptions extends PipelineOptions {
  maxAnswerLength?: number;
  maxQuestionLength?: number;
  topK?: number;
  threshold?: number;
  handleImpossible?: boolean;
}

export interface QuestionAnsweringResult extends PipelineResult {
  answer: string;
  score: number;
  start: number;
  end: number;
}

// ============================================================================
// Question Answering Pipeline
// ============================================================================

export class QuestionAnsweringPipeline extends BasePipeline<
  QAInput | QAInput[],
  QuestionAnsweringResult | QuestionAnsweringResult[]
> {
  private tokenizer: Tokenizer | null = null;
  private onnxModel: LoadedModel | null = null;
  private modelUrl: string;
  private tokenizerUrl: string;

  constructor(config?: PipelineConfig) {
    super(config ?? {
      task: 'question-answering',
      model: 'default',
    });
    this.modelUrl = (config?.model && config.model !== 'default') ? config.model : DEFAULT_MODELS.model;
    this.tokenizerUrl = DEFAULT_MODELS.tokenizer;
  }

  override async initialize(): Promise<void> {
    await super.initialize();

    if (!this.tokenizer) {
      this.tokenizer = await Tokenizer.fromUrl(this.tokenizerUrl);
    }

    if (!this.onnxModel) {
      const modelData = await loadModelData(this.modelUrl, { cache: this.config.cache ?? true });
      this.onnxModel = await loadModelFromBuffer(modelData);
    }
  }

  setTokenizer(tokenizer: Tokenizer): void {
    this.tokenizer = tokenizer;
  }

  override async run(
    input: QAInput | QAInput[],
    options?: QuestionAnsweringOptions
  ): Promise<QuestionAnsweringResult | QuestionAnsweringResult[]> {
    await this.initialize();

    const inputs = Array.isArray(input) ? input : [input];
    const results = await Promise.all(
      inputs.map(i => this.answerQuestion(i, options ?? {}))
    );

    return Array.isArray(input) ? results : results[0]!;
  }

  private async answerQuestion(
    input: QAInput,
    options: QuestionAnsweringOptions
  ): Promise<QuestionAnsweringResult> {
    const startTime = performance.now();
    const { question, context } = input;
    const maxAnswerLength = options.maxAnswerLength ?? 30;

    // No padding — QA runs one example at a time and padding wastes compute
    const encoded = this.tokenizer!.encode(question, {
      textPair: context,
      addSpecialTokens: true,
      maxLength: 512,
      truncation: true,
      padding: 'do_not_pad',
      returnAttentionMask: true,
      returnTokenTypeIds: true,
    });

    const seqLen = encoded.inputIds.length;

    const inputIds = new EdgeFlowTensor(
      BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))),
      [1, seqLen],
      'int64'
    );
    const attentionMask = new EdgeFlowTensor(
      BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))),
      [1, seqLen],
      'int64'
    );

    const namedInputs = new Map<string, EdgeFlowTensor>();
    namedInputs.set('input_ids', inputIds);
    namedInputs.set('attention_mask', attentionMask);

    const outputs = await runInferenceNamed(this.onnxModel!, namedInputs);

    if (outputs.length < 2) {
      return { answer: '', score: 0, start: 0, end: 0, processingTime: performance.now() - startTime };
    }

    const startLogits = (outputs[0] as EdgeFlowTensor).toFloat32Array();
    const endLogits = (outputs[1] as EdgeFlowTensor).toFloat32Array();

    const startProbs = softmax(new EdgeFlowTensor(new Float32Array(startLogits), [seqLen], 'float32')).toFloat32Array();
    const endProbs = softmax(new EdgeFlowTensor(new Float32Array(endLogits), [seqLen], 'float32')).toFloat32Array();

    // Constrain answer span to the context portion only (tokenTypeIds === 1).
    // tokenTypeIds: 0 = question tokens ([CLS], question, [SEP]), 1 = context tokens.
    const typeIds = encoded.tokenTypeIds ?? new Array(seqLen).fill(1);
    // Find where context starts (first index with typeId === 1)
    const contextStart = typeIds.findIndex(t => t === 1);
    const spanStart = contextStart >= 0 ? contextStart : 0;
    const spanEnd = seqLen - 1; // last non-padding position

    let bestStartIdx = spanStart;
    let bestEndIdx = spanStart;
    let bestScore = -Infinity;

    for (let s = spanStart; s <= spanEnd; s++) {
      for (let e = s; e < Math.min(s + maxAnswerLength, spanEnd + 1); e++) {
        const score = (startProbs[s] ?? 0) * (endProbs[e] ?? 0);
        if (score > bestScore) {
          bestScore = score;
          bestStartIdx = s;
          bestEndIdx = e;
        }
      }
    }

    // Decode the answer span directly from token IDs in the context portion
    const answerTokenIds = encoded.inputIds.slice(bestStartIdx, bestEndIdx + 1);
    const answer = this.tokenizer!.decode(answerTokenIds, true);

    return {
      answer: answer || '',
      score: Math.max(0, bestScore),
      start: bestStartIdx,
      end: bestEndIdx,
      processingTime: performance.now() - startTime,
    };
  }


  protected async preprocess(input: QAInput | QAInput[]): Promise<EdgeFlowTensor[]> {
    const qaInput = Array.isArray(input) ? input[0]! : input;
    const encoded = this.tokenizer!.encode(qaInput.question, {
      textPair: qaInput.context,
      addSpecialTokens: true,
      maxLength: 512,
      truncation: true,
      returnAttentionMask: true,
      returnTokenTypeIds: true,
    });

    return [
      new EdgeFlowTensor(
        BigInt64Array.from(encoded.inputIds.map(id => BigInt(id))),
        [1, encoded.inputIds.length],
        'int64'
      ),
      new EdgeFlowTensor(
        BigInt64Array.from(encoded.attentionMask.map(m => BigInt(m))),
        [1, encoded.attentionMask.length],
        'int64'
      ),
    ];
  }

  protected async postprocess(
    outputs: EdgeFlowTensor[],
    _options?: PipelineOptions
  ): Promise<QuestionAnsweringResult | QuestionAnsweringResult[]> {
    if (outputs.length < 2) {
      return { answer: '', score: 0, start: 0, end: 0 };
    }

    const startLogits = outputs[0]!.toFloat32Array();
    const endLogits = outputs[1]!.toFloat32Array();
    const seqLen = startLogits.length;

    const startProbs = softmax(new EdgeFlowTensor(startLogits, [seqLen], 'float32')).toFloat32Array();
    const endProbs = softmax(new EdgeFlowTensor(endLogits, [seqLen], 'float32')).toFloat32Array();

    let bestStart = 0;
    let bestEnd = 0;
    let bestScore = 0;

    for (let start = 0; start < seqLen; start++) {
      for (let end = start; end < Math.min(start + 30, seqLen); end++) {
        const score = (startProbs[start] ?? 0) * (endProbs[end] ?? 0);
        if (score > bestScore) {
          bestScore = score;
          bestStart = start;
          bestEnd = end;
        }
      }
    }

    return {
      answer: '',
      score: bestScore,
      start: bestStart,
      end: bestEnd,
    };
  }
}

// ============================================================================
// Factory
// ============================================================================

export function createQuestionAnsweringPipeline(
  config?: PipelineConfig
): QuestionAnsweringPipeline {
  return new QuestionAnsweringPipeline(config);
}

registerPipeline('question-answering', (config) => new QuestionAnsweringPipeline(config));
