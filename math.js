/**
 * math.js
 * Shared low-level math utilities for the entire chat-AI pipeline.
 * No external dependencies — pure JS.
 *
 * Covers:
 *   - Vector / matrix operations
 *   - Activations + their derivatives  (sigmoid, tanh, relu, softmax)
 *   - Weight initializers              (xavier, he, zeros, ones)
 *   - Numerical helpers                (clip, sample, argmax)
 */

// ─── Vector ops ───────────────────────────────────────────────────────────────

/** Dot product  a · b */
export function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Element-wise add  a + b  (new array) */
export function vadd(a, b) {
  return a.map((v, i) => v + b[i]);
}

/** Element-wise subtract  a - b  (new array) */
export function vsub(a, b) {
  return a.map((v, i) => v - b[i]);
}

/** Element-wise multiply  a ⊙ b  (Hadamard product, new array) */
export function vmul(a, b) {
  return a.map((v, i) => v * b[i]);
}

/** Scalar multiply  s * a  (new array) */
export function vscale(a, s) {
  return a.map(v => v * s);
}

/** L2 norm  ‖a‖ */
export function vnorm(a) {
  return Math.sqrt(dot(a, a));
}

/** Clip each element to [min, max] */
export function vclip(a, min, max) {
  return a.map(v => Math.min(max, Math.max(min, v)));
}

/** Zero vector of length n */
export function zeros(n) {
  return new Float64Array(n);
}

/** Ones vector of length n */
export function ones(n) {
  return new Float64Array(n).fill(1);
}

/** Deep copy a Float64Array or plain array */
export function vcopy(a) {
  return Float64Array.from(a);
}

// ─── Matrix ops  (row-major flat arrays) ─────────────────────────────────────
// Matrix M of shape [rows × cols] stored as M[r*cols + c]

/** Create a zero matrix [rows × cols] */
export function matZeros(rows, cols) {
  return new Float64Array(rows * cols);
}

/**
 * Matrix-vector multiply  y = M · x
 * M: [rows × cols] flat,  x: [cols],  returns [rows]
 */
export function matvec(M, x, rows, cols) {
  const y = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let s = 0;
    for (let c = 0; c < cols; c++) s += M[r * cols + c] * x[c];
    y[r] = s;
  }
  return y;
}

/**
 * Outer product  u ⊗ v  → [m × n] flat matrix
 * Used to compute weight gradients: dW += delta ⊗ input
 */
export function outer(u, v) {
  const m = u.length, n = v.length;
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i++)
    for (let j = 0; j < n; j++)
      out[i * n + j] = u[i] * v[j];
  return out;
}

/**
 * Accumulate  dst += src  (in-place, both Float64Array same length)
 */
export function accum(dst, src) {
  for (let i = 0; i < dst.length; i++) dst[i] += src[i];
}

// ─── Activations ─────────────────────────────────────────────────────────────

/** sigmoid(x) = 1 / (1 + e^-x) */
export function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}
/** sigmoid applied element-wise */
export function vsigmoid(a) {
  return a.map(sigmoid);
}
/** d/dx sigmoid(x) = sigmoid(x)(1 - sigmoid(x)) */
export function sigmoidGrad(sigmoidOut) {
  return sigmoidOut.map(s => s * (1 - s));
}

/** tanh applied element-wise */
export function vtanh(a) {
  return a.map(Math.tanh);
}
/** d/dx tanh(x) = 1 - tanh(x)^2 */
export function tanhGrad(tanhOut) {
  return tanhOut.map(t => 1 - t * t);
}

/** ReLU applied element-wise */
export function vrelu(a) {
  return a.map(v => Math.max(0, v));
}
/** d/dx ReLU(x) = 1 if x > 0 else 0  (using pre-activation) */
export function reluGrad(preAct) {
  return preAct.map(v => (v > 0 ? 1 : 0));
}

/**
 * Softmax — numerically stable via max subtraction.
 * Converts raw logits → probability distribution.
 */
export function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}

// ─── Weight initializers ──────────────────────────────────────────────────────

/**
 * Xavier / Glorot uniform  ∈ [ -√(6/(fin+fout)), +√(6/(fin+fout)) ]
 * Best for sigmoid / tanh activations.
 */
export function xavierUniform(size, fanIn, fanOut) {
  const limit = Math.sqrt(6 / (fanIn + fanOut));
  return Float64Array.from({ length: size }, () => (Math.random() * 2 - 1) * limit);
}

/**
 * He / Kaiming uniform  ∈ [ -√(6/fin), +√(6/fin) ]
 * Best for ReLU activations.
 */
export function heUniform(size, fanIn) {
  const limit = Math.sqrt(6 / fanIn);
  return Float64Array.from({ length: size }, () => (Math.random() * 2 - 1) * limit);
}

// ─── Numerical helpers ────────────────────────────────────────────────────────

/** Index of maximum value in array */
export function argmax(a) {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
  return best;
}

/**
 * Temperature sampling — draw one index from a probability distribution.
 * temperature=1.0 → normal sampling
 * temperature<1.0 → more confident / deterministic
 * temperature>1.0 → more random / creative
 */
export function sample(probs, temperature = 1.0) {
  // Apply temperature to logits (re-derive from probs then re-softmax)
  if (temperature !== 1.0) {
    const logits = probs.map(p => Math.log(p + 1e-10) / temperature);
    probs = softmax(logits);
  }
  let r = Math.random();
  for (let i = 0; i < probs.length; i++) {
    r -= probs[i];
    if (r <= 0) return i;
  }
  return probs.length - 1;
}

/**
 * Cross-entropy loss for one timestep.
 * targetId — the correct token index
 * probs    — softmax output [vocabSize]
 */
export function crossEntropy(probs, targetId) {
  return -Math.log(probs[targetId] + 1e-10);
}

/**
 * Gradient of cross-entropy + softmax combined (very clean result):
 * dL/dlogit[i] = probs[i] - 1  if i == target
 *              = probs[i]       otherwise
 */
export function crossEntropyGrad(probs, targetId) {
  const grad = Float64Array.from(probs);
  grad[targetId] -= 1;
  return grad;
}

/**
 * Gradient clipping by global norm.
 * Prevents exploding gradients in RNNs.
 * Returns scaled copies of all grad arrays.
 *
 * @param {Float64Array[]} grads
 * @param {number}         maxNorm
 * @returns {Float64Array[]}
 */
export function clipGradsByNorm(grads, maxNorm = 5.0) {
  let totalNorm = 0;
  for (const g of grads) totalNorm += dot(g, g);
  totalNorm = Math.sqrt(totalNorm);

  if (totalNorm <= maxNorm) return grads;
  const ratio = maxNorm / totalNorm;
  return grads.map(g => vscale(g, ratio));
}