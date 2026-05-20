import fs   from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { Tokenizer, SPECIAL_TOKENS }  from "./tokenizer.js";
import { EmbeddingLayer }              from "./embedding.js";
import { GRUEncoder }                  from "./encoder.js";
import { AttentionLayer }              from "./attention.js";
import { GRUDecoder }                  from "./decoder.js";
import {
  zeros, vcopy, vadd, vscale, accum,
  argmax, sample, softmax,
  crossEntropy, crossEntropyGrad,
  matvec, outer,
  vsigmoid, vtanh, sigmoidGrad, tanhGrad, vmul,
  clipGradsByNorm,
} from "./math.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Hyper-parameters ─────────────────────────────────────────────────────────

export const DEFAULTS = {
  embedDim:    64,
  hiddenDim:   128,
  attnDim:     64,
  maxDecLen:   40,
  temperature: 0.8,
  learningRate: 0.001,
  gradClip:    5.0,
};

// ─── Adam optimizer ───────────────────────────────────────────────────────────
// t increments once per call to step(), not once per weight element.

class Adam {
  constructor(lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8) {
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps;
    this.m  = new Map();
    this.v  = new Map();
    this.t  = new Map(); // per-key step counter
  }

  step(key, w, g) {
    if (!this.m.has(key)) {
      this.m.set(key, new Float64Array(w.length));
      this.v.set(key, new Float64Array(w.length));
      this.t.set(key, 0);
    }
    const m  = this.m.get(key);
    const v  = this.v.get(key);
    const t  = this.t.get(key) + 1;
    this.t.set(key, t);

    const bc1 = 1 - Math.pow(this.beta1, t);
    const bc2 = 1 - Math.pow(this.beta2, t);

    for (let i = 0; i < w.length; i++) {
      if (!isFinite(g[i])) continue;            // skip NaN/Inf elements
      m[i] = this.beta1 * m[i] + (1 - this.beta1) * g[i];
      v[i] = this.beta2 * v[i] + (1 - this.beta2) * g[i] * g[i];
      w[i] -= this.lr * (m[i] / bc1) / (Math.sqrt(v[i] / bc2) + this.eps);
    }
  }
}

// ─── ChatModel ────────────────────────────────────────────────────────────────

export class ChatModel {
  constructor(vocabSize, maxSeqLen, cfg = {}) {
    const c = { ...DEFAULTS, ...cfg };
    this.cfg       = c;
    this.vocabSize = vocabSize;
    this.maxSeqLen = maxSeqLen;

    this.embedding = new EmbeddingLayer(vocabSize, c.embedDim, maxSeqLen,
                                        { usePositional: true });
    this.encoder   = new GRUEncoder(c.embedDim, c.hiddenDim);
    this.attention = new AttentionLayer(c.hiddenDim, c.hiddenDim, c.attnDim);
    this.decoder   = new GRUDecoder(c.embedDim, c.hiddenDim, c.hiddenDim, vocabSize);
    this.optimizer = new Adam(c.learningRate);

    const params = this._countParams();
    console.log(`🤖 ChatModel  vocab=${vocabSize}  embed=${c.embedDim}  hidden=${c.hiddenDim}  params≈${params.toLocaleString()}`);
  }

  _countParams() {
    const { embedDim: E, hiddenDim: H, attnDim: A } = this.cfg;
    const V = this.vocabSize;
    return V*E + 2*(3*(H*(E+H)+H)) + (A*H*2+A) + V*H+V;
  }

  // ── trainStep ─────────────────────────────────────────────────────────────
  //
  // One complete forward → backward → update cycle for a single Q&A pair.
  // All caches are populated during the forward pass and consumed ONCE
  // during backward — no re-forwards.

  trainStep(qIds, aIds) {
    if (aIds.length < 2) return null;

    // ── Zero grads ──────────────────────────────────────────────────────────
    this.embedding.zeroGrad();
    this.encoder.zeroGrad();
    this.attention.zeroGrad();
    this.decoder.zeroGrad();
    this.decoder.resetSteps();

    // ── FORWARD ─────────────────────────────────────────────────────────────

    // 1. Embed question
    const qEmbs = this.embedding.forward(qIds);

    // 2. Encode question  →  encStates[t], finalHidden
    const { hiddenStates: encStates, finalHidden } = this.encoder.forward(qEmbs);

    // 3. Decode with teacher forcing
    //    Input  tokens: aIds[0 .. T-2]   (starts with <A>)
    //    Target tokens: aIds[1 .. T-1]   (ends   with <EOS>)
    const decInputIds = aIds.slice(0, -1);
    const targetIds   = aIds.slice(1);
    const T = decInputIds.length;

    // Store per-step caches for clean backward
    const stepCache = [];   // { decEmb, context, attnWeights, decHiddenIn }

    let decHidden = vcopy(finalHidden);

    for (let t = 0; t < T; t++) {
      const decEmb = this.embedding.forward([decInputIds[t]])[0];

      // Save the decoder hidden state BEFORE this step (needed for attn backward)
      const decHiddenIn = vcopy(decHidden);

      const { context, weights: attnWeights } = this.attention.forward(encStates, decHidden);
      const { h } = this.decoder.step(decEmb, context, decHidden);

      stepCache.push({ decEmb, context, attnWeights, decHiddenIn });
      decHidden = h;
    }

    // ── BACKWARD ────────────────────────────────────────────────────────────

    // 4. Decoder backward  (handles output projection + GRU cell)
    const { loss, dContexts, dEmbeddings: dDecEmbs, dInitHidden } =
      this.decoder.backward(targetIds);

    if (!isFinite(loss)) return NaN;

    // 5. Attention backward for each timestep (reverse order)
    //    Accumulate gradients into encoder states
    const dEncStatesTotal = encStates.map(() => zeros(this.cfg.hiddenDim));
    let dDecHiddenFromAttn = zeros(this.cfg.hiddenDim);

    for (let t = T - 1; t >= 0; t--) {
      // Restore attention cache by re-running forward with the EXACT
      // hidden states we saved during the forward pass
      this.attention.forward(encStates, stepCache[t].decHiddenIn);

      const { dEncStates, dDecHidden } = this.attention.backward(dContexts[t]);

      for (let s = 0; s < encStates.length; s++) {
        accum(dEncStatesTotal[s], dEncStates[s]);
      }
      accum(dDecHiddenFromAttn, dDecHidden);
    }

    // Combine attention's grad on decoder hidden with decoder's dInitHidden
    const dFinalHidden = vadd(dInitHidden, dDecHiddenFromAttn);

    // 6. Encoder backward (BPTT)
    const dQEmbs = this.encoder.backward(dEncStatesTotal, dFinalHidden);

    // 7. Embedding backward
    this.embedding.backward(dQEmbs);       // question embeddings
    this.embedding.backward(dDecEmbs);     // answer input embeddings

    // ── OPTIMIZER STEP ───────────────────────────────────────────────────────
    this._optimizerStep();

    return loss;
  }

  _optimizerStep() {
    const clip = this.cfg.gradClip;

    // Embedding (sparse rows)
    for (const { id, grad } of this.embedding.table.sparseGradients()) {
      const clipped = clipGradsByNorm([Float64Array.from(grad)], clip)[0];
      const offset  = id * this.cfg.embedDim;
      const wSlice  = this.embedding.table.weights.slice(offset, offset + this.cfg.embedDim);
      this.optimizer.step(`emb_${id}`, wSlice, clipped);
      for (let d = 0; d < this.cfg.embedDim; d++) {
        this.embedding.table.weights[offset + d] = wSlice[d];
      }
    }

    // Encoder, Attention, Decoder — dense params
    const modules = [
      { name: 'enc',  params: this.encoder.parameters()   },
      { name: 'attn', params: this.attention.parameters()  },
      { name: 'dec',  params: this.decoder.parameters()    },
    ];

    for (const { name, params } of modules) {
      let idx = 0;
      for (const { w, g } of params) {
        const clipped = clipGradsByNorm([g], clip)[0];
        this.optimizer.step(`${name}_${idx++}`, w, clipped);
      }
    }
  }

  // ── generate (inference) ──────────────────────────────────────────────────

  generate(qIds, { greedy = false } = {}) {
    const { maxDecLen, temperature } = this.cfg;

    const qEmbs = this.embedding.forward(qIds);
    const { hiddenStates: encStates, finalHidden } = this.encoder.forward(qEmbs);

    let decHidden = vcopy(finalHidden);
    let prevId    = SPECIAL_TOKENS["<A>"];
    const output  = [];

    for (let t = 0; t < maxDecLen; t++) {
      const decEmb = this.embedding.forward([prevId])[0];
      const { context } = this.attention.forward(encStates, decHidden);
      const { h, probs } = this.decoder.step(decEmb, context, decHidden);

      decHidden = h;
      const nextId = greedy
        ? argmax(Array.from(probs))
        : sample(Array.from(probs), temperature);

      if (nextId === SPECIAL_TOKENS["<EOS>"]) break;
      if (nextId === SPECIAL_TOKENS["<PAD>"]) break;
      output.push(nextId);
      prevId = nextId;
    }

    this.decoder.resetSteps();
    return output;
  }

  // ── save / load ───────────────────────────────────────────────────────────

  save(dir) {
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    this.embedding.save(path.join(dir, "embedding_weights.json"));
    this.encoder.save(path.join(dir,   "encoder_weights.json"));
    this.attention.save(path.join(dir, "attention_weights.json"));
    this.decoder.save(path.join(dir,   "decoder_weights.json"));
    fs.writeFileSync(path.join(dir, "model_cfg.json"),
      JSON.stringify({ vocabSize: this.vocabSize, maxSeqLen: this.maxSeqLen, cfg: this.cfg }));
    console.log(`✅ Model saved → ${dir}/`);
  }

  load(dir) {
    this.embedding.load(path.join(dir, "embedding_weights.json"));
    this.encoder.load(path.join(dir,   "encoder_weights.json"));
    this.attention.load(path.join(dir, "attention_weights.json"));
    this.decoder.load(path.join(dir,   "decoder_weights.json"));
    console.log(`✅ Model loaded ← ${dir}/`);
  }

  static fromSaved(dir) {
    const cfg   = JSON.parse(fs.readFileSync(path.join(dir, "model_cfg.json"), "utf-8"));
    const model = new ChatModel(cfg.vocabSize, cfg.maxSeqLen, cfg.cfg);
    model.load(dir);
    return model;
  }
}