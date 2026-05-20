/**
 * embedding.js
 * Embedding layer — converts token IDs → dense float vectors.
 *
 * What's inside:
 *   EmbeddingTable    — learnable weight matrix  [vocabSize × embedDim]
 *   PositionalEncoder — sinusoidal position signal added to embeddings
 *   EmbeddingLayer    — combines both; main entry point for the model
 *
 * Math recap:
 *   embed(id)  =  W[id]            (row lookup, O(1))
 *   out[pos]   =  W[id] + PE[pos]  (add position signal)
 *
 * Gradient note:
 *   Only the rows that were actually looked up accumulate gradients.
 *   backward() does a sparse update: grad flows back through each
 *   looked-up row, which is exactly what an optimizer (SGD/Adam) needs.
 */

import fs   from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Tokenizer, SPECIAL_TOKENS } from "./tokenizer.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Tiny math helpers (no external deps) ────────────────────────────────────

/** Dot product of two same-length arrays. */
function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Element-wise add (returns new array). */
function add(a, b) {
  return a.map((v, i) => v + b[i]);
}

/** Scalar multiply (returns new array). */
function scale(a, s) {
  return a.map(v => v * s);
}

/** L2 norm of a vector. */
function norm(a) {
  return Math.sqrt(dot(a, a));
}

/** Cosine similarity between two vectors. */
function cosineSim(a, b) {
  const d = norm(a) * norm(b);
  return d === 0 ? 0 : dot(a, b) / d;
}

// ─── Weight initializers ──────────────────────────────────────────────────────

/**
 * Xavier / Glorot uniform initializer.
 * Keeps activations well-scaled at the start of training.
 * range = ±sqrt(6 / (fanIn + fanOut))
 */
function xavierUniform(fanIn, fanOut) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return Array.from({ length: fanIn * fanOut }, () =>
    (Math.random() * 2 - 1) * limit
  );
}

// ─── EmbeddingTable ───────────────────────────────────────────────────────────

/**
 * A learnable lookup table of shape [vocabSize × embedDim].
 * Each row is the vector representation of one token.
 *
 * The <PAD> row (id=0) is kept at zero and never updated — padding
 * tokens should contribute nothing to the model's computation.
 */
class EmbeddingTable {
  /**
   * @param {number} vocabSize   — total number of tokens
   * @param {number} embedDim    — vector width (e.g. 32, 64, 128)
   */
  constructor(vocabSize, embedDim) {
    this.vocabSize = vocabSize;
    this.embedDim  = embedDim;

    // Flat array, row-major: weights[id * embedDim + d]
    this.weights = xavierUniform(vocabSize, embedDim);

    // Zero out the PAD row so padding is invisible to the model
    this._zeroPadRow();

    // Gradient accumulator (same shape as weights, starts at zero)
    this.grads = new Float64Array(vocabSize * embedDim);

    // Track which token ids were used in the last forward pass (for sparse updates)
    this._lastIds = [];
  }

  _zeroPadRow() {
    const PAD = SPECIAL_TOKENS["<PAD>"]; // 0
    for (let d = 0; d < this.embedDim; d++) {
      this.weights[PAD * this.embedDim + d] = 0;
    }
  }

  // ── Forward ─────────────────────────────────────────────────────────────────

  /**
   * Look up a single token id → Float64Array of length embedDim.
   * @param {number} id
   * @returns {Float64Array}
   */
  lookup(id) {
    const start  = id * this.embedDim;
    return new Float64Array(this.weights.slice(start, start + this.embedDim));
  }

  /**
   * Look up a sequence of token ids → 2-D array [seqLen × embedDim].
   * @param {number[]} ids
   * @returns {Float64Array[]}
   */
  forward(ids) {
    this._lastIds = ids;
    return ids.map(id => this.lookup(id));
  }

  // ── Backward ─────────────────────────────────────────────────────────────────

  /**
   * Accumulate gradients for the rows that were looked up.
   * dLoss/dW[id] += upstreamGrad[pos]  for each (pos, id) pair.
   *
   * @param {Float64Array[]} upstreamGrads  — same shape as forward output [seqLen × embedDim]
   */
  backward(upstreamGrads) {
    for (let pos = 0; pos < this._lastIds.length; pos++) {
      const id = this._lastIds[pos];
      if (id === SPECIAL_TOKENS["<PAD>"]) continue; // never update PAD

      const offset = id * this.embedDim;
      const g      = upstreamGrads[pos];
      for (let d = 0; d < this.embedDim; d++) {
        this.grads[offset + d] += g[d];
      }
    }
  }

  /** Zero out accumulated gradients (call before each new batch). */
  zeroGrad() {
    this.grads.fill(0);
  }

  // ── Parameter access (for optimizer) ────────────────────────────────────────

  /**
   * Returns a list of { id, grad } for every token that received a gradient.
   * Optimizers use this for sparse weight updates.
   */
  sparseGradients() {
    const seen = [...new Set(this._lastIds)].filter(
      id => id !== SPECIAL_TOKENS["<PAD>"]
    );
    return seen.map(id => {
      const offset = id * this.embedDim;
      return {
        id,
        grad: Array.from(this.grads.subarray(offset, offset + this.embedDim)),
      };
    });
  }

  /**
   * Apply an SGD step directly on the sparse rows.
   * lr = learning rate.
   */
  sgdStep(lr = 0.01) {
    for (const { id, grad } of this.sparseGradients()) {
      const offset = id * this.embedDim;
      for (let d = 0; d < this.embedDim; d++) {
        this.weights[offset + d] -= lr * grad[d];
      }
    }
  }
}

// ─── PositionalEncoder ────────────────────────────────────────────────────────

/**
 * Fixed (non-learnable) sinusoidal positional encoding.
 *
 * Formula (Vaswani et al., 2017):
 *   PE[pos, 2i]   = sin(pos / 10000^(2i/d))
 *   PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
 *
 * Why sinusoidal?
 *  - No parameters to learn → works out of the box on small datasets
 *  - Each position gets a unique fingerprint
 *  - The model can generalize to sequences longer than those in training
 */
class PositionalEncoder {
  /**
   * @param {number} maxSeqLen  — longest sequence you'll ever encode
   * @param {number} embedDim   — must match EmbeddingTable.embedDim
   */
  constructor(maxSeqLen, embedDim) {
    this.maxSeqLen = maxSeqLen;
    this.embedDim  = embedDim;
    this.table     = this._build(maxSeqLen, embedDim);
  }

  _build(maxLen, dim) {
    const table = [];
    for (let pos = 0; pos < maxLen; pos++) {
      const row = new Float64Array(dim);
      for (let i = 0; i < dim / 2; i++) {
        const angle = pos / Math.pow(10000, (2 * i) / dim);
        row[2 * i]     = Math.sin(angle);
        row[2 * i + 1] = Math.cos(angle);
      }
      table.push(row);
    }
    return table;
  }

  /**
   * Return the positional vector for a given position.
   * @param {number} pos
   * @returns {Float64Array}
   */
  get(pos) {
    if (pos >= this.maxSeqLen) {
      throw new RangeError(`Position ${pos} exceeds maxSeqLen ${this.maxSeqLen}`);
    }
    return this.table[pos];
  }

  /**
   * Add positional encodings to a sequence of embeddings (in-place).
   * @param {Float64Array[]} embeddings  [seqLen × embedDim]
   * @returns {Float64Array[]}           same array, mutated
   */
  encode(embeddings) {
    for (let pos = 0; pos < embeddings.length; pos++) {
      const pe = this.get(pos);
      for (let d = 0; d < this.embedDim; d++) {
        embeddings[pos][d] += pe[d];
      }
    }
    return embeddings;
  }
}

// ─── EmbeddingLayer ───────────────────────────────────────────────────────────

/**
 * The full embedding pipeline used by the model:
 *   token ids → token embeddings → + positional encoding → output
 *
 * Usage:
 *   const emb = new EmbeddingLayer(vocabSize, embedDim, maxSeqLen);
 *   const vectors = emb.forward([4, 12, 7, 3]);   // ids → Float64Array[]
 *   emb.backward(upstreamGrads);                   // backprop
 *   emb.table.sgdStep(0.01);                       // weight update
 */
class EmbeddingLayer {
  /**
   * @param {number} vocabSize
   * @param {number} embedDim    recommended: 32–128 for small chat models
   * @param {number} maxSeqLen   set to your batchEncode maxLen
   * @param {object} [opts]
   * @param {boolean} [opts.usePositional=true]  toggle positional encoding
   */
  constructor(vocabSize, embedDim, maxSeqLen, { usePositional = true } = {}) {
    this.embedDim      = embedDim;
    this.usePositional = usePositional;

    this.table    = new EmbeddingTable(vocabSize, embedDim);
    this.posEnc   = usePositional
      ? new PositionalEncoder(maxSeqLen, embedDim)
      : null;
  }

  // ── Forward ─────────────────────────────────────────────────────────────────

  /**
   * Convert a sequence of token ids to a matrix of dense vectors.
   *
   * @param {number[]} ids        token id sequence
   * @returns {Float64Array[]}    [seqLen × embedDim] — one vector per token
   */
  forward(ids) {
    const embeddings = this.table.forward(ids);   // token vectors
    if (this.usePositional) this.posEnc.encode(embeddings);  // + position
    return embeddings;
  }

  /**
   * Batch forward: process multiple sequences at once.
   * All sequences must be the same length (pad with <PAD> if needed).
   *
   * @param {number[][]} batch   [batchSize × seqLen]
   * @returns {Float64Array[][]} [batchSize × seqLen × embedDim]
   */
  forwardBatch(batch) {
    return batch.map(ids => this.forward(ids));
  }

  // ── Backward ─────────────────────────────────────────────────────────────────

  /**
   * Backprop through the embedding lookup.
   * Positional encoding is fixed (no grad needed there).
   *
   * @param {Float64Array[]} upstreamGrads  [seqLen × embedDim]
   */
  backward(upstreamGrads) {
    this.table.backward(upstreamGrads);
  }

  /** Zero all accumulated gradients. Call before each new forward pass. */
  zeroGrad() {
    this.table.zeroGrad();
  }

  // ── Utilities ────────────────────────────────────────────────────────────────

  /**
   * Cosine similarity between two token ids (after training, similar words
   * cluster together in embedding space).
   */
  similarity(idA, idB) {
    return cosineSim(
      Array.from(this.table.lookup(idA)),
      Array.from(this.table.lookup(idB))
    );
  }

  /**
   * Find the N most similar tokens to a given id (nearest neighbours).
   * Useful for sanity-checking learned representations.
   *
   * @param {number}   id
   * @param {number}   [n=5]
   * @param {object}   [reverseVocab]  optional id→word map for labels
   * @returns {{ id: number, word: string|null, similarity: number }[]}
   */
  nearestNeighbours(id, n = 5, reverseVocab = {}) {
    const target = Array.from(this.table.lookup(id));
    const scores = [];

    for (let otherId = 0; otherId < this.table.vocabSize; otherId++) {
      if (otherId === id) continue;
      scores.push({
        id:         otherId,
        word:       reverseVocab[otherId] ?? null,
        similarity: cosineSim(target, Array.from(this.table.lookup(otherId))),
      });
    }

    return scores
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, n);
  }

  // ── Serialization ─────────────────────────────────────────────────────────

  /** Save weights to disk. */
  save(filePath) {
    const data = {
      vocabSize:     this.table.vocabSize,
      embedDim:      this.embedDim,
      usePositional: this.usePositional,
      weights:       Array.from(this.table.weights),
    };
    fs.writeFileSync(filePath, JSON.stringify(data));
    console.log(`💾 Embedding weights saved → ${filePath}`);
  }

  /** Load weights from disk. */
  load(filePath) {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    if (data.vocabSize !== this.table.vocabSize || data.embedDim !== this.embedDim) {
      throw new Error(
        `Shape mismatch: file has [${data.vocabSize}×${data.embedDim}], ` +
        `layer expects [${this.table.vocabSize}×${this.embedDim}]`
      );
    }
    this.table.weights = data.weights;
    console.log(`📂 Embedding weights loaded ← ${filePath}`);
  }
}

// ─── Demo ─────────────────────────────────────────────────────────────────────

function main() {
  // 1. Load vocab produced by tokenizer.js
  const vocabPath = path.join(__dirname, "vocab.json");
  if (!fs.existsSync(vocabPath)) {
    console.error("❌ vocab.json not found — run tokenizer.js first");
    process.exit(1);
  }

  const tokenizer = new Tokenizer();
  tokenizer.loadVocab(vocabPath);

  // 2. Load encoded dataset
  const encPath = path.join(__dirname, "encoded_dataset.json");
  if (!fs.existsSync(encPath)) {
    console.error("❌ encoded_dataset.json not found — run tokenizer.js first");
    process.exit(1);
  }
  const { encoded, maxLen } = JSON.parse(fs.readFileSync(encPath, "utf-8"));

  // 3. Build embedding layer
  const EMBED_DIM = 64;  // ← tune this: 32 for tiny, 128 for richer
  const emb = new EmbeddingLayer(tokenizer.vocabSize, EMBED_DIM, maxLen);

  console.log("\n── EmbeddingLayer config ──────────────────────────────");
  console.log(`  vocab size  : ${tokenizer.vocabSize}`);
  console.log(`  embed dim   : ${EMBED_DIM}`);
  console.log(`  max seq len : ${maxLen}`);
  console.log(`  total params: ${tokenizer.vocabSize * EMBED_DIM}`);

  // 4. Forward pass on the first sequence
  const firstSeq  = encoded[0];
  const vectors   = emb.forward(firstSeq);

  console.log("\n── Forward pass (sequence 0) ──────────────────────────");
  console.log(`  Input IDs   : [${firstSeq.slice(0, 8).join(", ")} …]`);
  console.log(`  Output shape: ${vectors.length} × ${vectors[0].length}`);
  console.log(`  Token 0 vec : [${Array.from(vectors[0]).slice(0, 6).map(v => v.toFixed(4)).join(", ")} …]`);
  console.log(`  Token 1 vec : [${Array.from(vectors[1]).slice(0, 6).map(v => v.toFixed(4)).join(", ")} …]`);

  // 5. Simulated backward pass (upstream grad = all ones — just for shape demo)
  const fakeGrad = vectors.map(() => new Float64Array(EMBED_DIM).fill(0.01));
  emb.backward(fakeGrad);
  console.log("\n── Backward pass ──────────────────────────────────────");
  console.log(`  Sparse grad rows: ${emb.table.sparseGradients().length} tokens updated`);

  // 6. SGD step
  emb.table.sgdStep(0.01);
  console.log("  SGD step applied ✅");

  // 7. Nearest neighbours (random-init, so values are meaningless — 
  //    but after training these will reflect semantic similarity)
  const testId  = firstSeq[2]; // third token in sequence
  const testWord = tokenizer.reverseVocab[testId] ?? "?";
  const neighbours = emb.nearestNeighbours(testId, 3, tokenizer.reverseVocab);
  console.log(`\n── Nearest neighbours of "${testWord}" (id=${testId}) ──`);
  for (const n of neighbours) {
    console.log(`  "${n.word ?? n.id}"  sim=${n.similarity.toFixed(4)}`);
  }

  // 8. Batch forward on full dataset
  const batchVectors = emb.forwardBatch(encoded);
  console.log(`\n── Batch forward ──────────────────────────────────────`);
  console.log(`  Processed ${batchVectors.length} sequences`);
  console.log(`  Each: ${batchVectors[0].length} tokens × ${EMBED_DIM} dims`);

  // 9. Save weights
  emb.save(path.join(__dirname, "embedding_weights.json"));
}

main();

export { EmbeddingLayer, EmbeddingTable, PositionalEncoder, cosineSim, dot, add, scale };