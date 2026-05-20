/**
 * attention.js
 * Bahdanau (Additive) Attention Mechanism
 *
 * "Attention" answers the question:
 *   "When generating word t of the answer, which words in the question
 *    should I focus on?"
 *
 * For each decoder step, it:
 *   1. Scores every encoder hidden state against the current decoder state
 *   2. Normalises scores → weights (sum to 1) via softmax
 *   3. Returns a weighted sum of encoder states = "context vector"
 *
 * Bahdanau score:
 *   e[t,s] = v · tanh(Wa·h_enc[s] + Wb·h_dec)
 *   α[t,s] = softmax(e[t,:])
 *   c[t]   = Σ_s  α[t,s] · h_enc[s]
 *
 * Why additive attention for a small model?
 *   - Learns a non-linear compatibility function (more expressive than dot-product)
 *   - Works better when encoder and decoder have different hidden sizes
 *   - Well-tested on small datasets (original machine-translation paper used ~150k pairs)
 */

import fs from "fs";
import {
  zeros, vcopy, vadd, vscale,
  vtanh, tanhGrad, vmul,
  matvec, outer, accum,
  xavierUniform, matZeros,
  softmax, dot,
} from "./math.js";

// ─── AttentionLayer ───────────────────────────────────────────────────────────

export class AttentionLayer {
  /**
   * @param {number} encHiddenDim   — encoder hidden size
   * @param {number} decHiddenDim   — decoder hidden size
   * @param {number} [attnDim=64]   — internal attention dimension (v size)
   */
  constructor(encHiddenDim, decHiddenDim, attnDim = 64) {
    this.encHiddenDim = encHiddenDim;
    this.decHiddenDim = decHiddenDim;
    this.attnDim      = attnDim;
    const A = attnDim;

    // Wa: project encoder hidden  [A × encHiddenDim]
    this.Wa = xavierUniform(A * encHiddenDim, encHiddenDim, A);
    // Wb: project decoder hidden  [A × decHiddenDim]
    this.Wb = xavierUniform(A * decHiddenDim, decHiddenDim, A);
    // v:  score vector            [A]
    this.v  = xavierUniform(A, A, 1);

    // Gradient accumulators
    this.dWa = matZeros(A, encHiddenDim);
    this.dWb = matZeros(A, decHiddenDim);
    this.dv  = zeros(A);

    // Cache for backward
    this._cache = null;
  }

  // ── Forward ─────────────────────────────────────────────────────────────────

  /**
   * Compute context vector for one decoder timestep.
   *
   * @param {Float64Array[]} encStates   all encoder hidden states [seqLen × encHiddenDim]
   * @param {Float64Array}   decHidden   current decoder hidden state [decHiddenDim]
   * @returns {{
   *   context:  Float64Array,   // weighted sum of encoder states [encHiddenDim]
   *   weights:  Float64Array,   // attention weights (sum=1) [seqLen]
   * }}
   */
  forward(encStates, decHidden) {
    const seqLen = encStates.length;
    const A = this.attnDim;

    // Project decoder state once (shared across all encoder positions)
    const Wb_dec = matvec(this.Wb, decHidden, A, this.decHiddenDim); // [A]

    // Score each encoder position
    const energies    = new Float64Array(seqLen);
    const tanhInputs  = [];   // save for backward
    const tanhOutputs = [];

    for (let s = 0; s < seqLen; s++) {
      const Wa_enc = matvec(this.Wa, encStates[s], A, this.encHiddenDim); // [A]
      const combined = vadd(Wa_enc, Wb_dec);                               // [A]
      const tOut     = vtanh(combined);                                    // [A]
      tanhInputs.push(combined);
      tanhOutputs.push(tOut);
      energies[s] = dot(this.v, tOut);                                     // scalar
    }

    // Softmax over energies → attention weights
    const weights = softmax(Array.from(energies));                         // [seqLen]

    // Context vector = weighted sum of encoder states
    const context = zeros(this.encHiddenDim);
    for (let s = 0; s < seqLen; s++) {
      for (let d = 0; d < this.encHiddenDim; d++) {
        context[d] += weights[s] * encStates[s][d];
      }
    }

    this._cache = { encStates, decHidden, Wb_dec, tanhInputs, tanhOutputs, weights, energies };
    return { context, weights: Float64Array.from(weights) };
  }

  // ── Backward ─────────────────────────────────────────────────────────────────

  /**
   * @param {Float64Array} dContext   upstream gradient w.r.t. context [encHiddenDim]
   * @returns {{
   *   dEncStates: Float64Array[],   // grad for each encoder hidden state
   *   dDecHidden: Float64Array,     // grad for decoder hidden state
   * }}
   */
  backward(dContext) {
    const { encStates, decHidden, Wb_dec, tanhInputs, tanhOutputs, weights } = this._cache;
    const seqLen = encStates.length;
    const A = this.attnDim;

    // ── Grad through weighted sum: context = Σ α[s] * h_enc[s] ──────────────
    // d_alpha[s] = dContext · h_enc[s]
    // d_h_enc[s] += alpha[s] * dContext
    const dAlpha    = new Float64Array(seqLen);
    const dEncStates = encStates.map(() => zeros(this.encHiddenDim));

    for (let s = 0; s < seqLen; s++) {
      dAlpha[s] = dot(dContext, encStates[s]);
      for (let d = 0; d < this.encHiddenDim; d++) {
        dEncStates[s][d] += weights[s] * dContext[d];
      }
    }

    // ── Grad through softmax ─────────────────────────────────────────────────
    // d_energy[s] = alpha[s] * (dAlpha[s] - Σ_k alpha[k]*dAlpha[k])
    const dotAlphaDAlpha = dot(Array.from(weights), Array.from(dAlpha));
    const dEnergies = new Float64Array(seqLen);
    for (let s = 0; s < seqLen; s++) {
      dEnergies[s] = weights[s] * (dAlpha[s] - dotAlphaDAlpha);
    }

    // ── Grad through score: e[s] = v · tanh(Wa*h_enc[s] + Wb*h_dec) ────────
    const dWb_dec = zeros(A); // accumulate decoder projection grads

    for (let s = 0; s < seqLen; s++) {
      // d_tanh_out[s] = dEnergies[s] * v
      const dTanhOut = vscale(this.v, dEnergies[s]);                 // [A]
      accum(this.dv, vmul(dTanhOut, tanhOutputs[s]));                 // dv += dEnergies[s] * tanh_out

      // Through tanh
      const dCombined = vmul(dTanhOut, tanhGrad(tanhOutputs[s]));    // [A]

      // Through Wa * h_enc[s]
      accum(this.dWa, outer(dCombined, encStates[s]));
      for (let d = 0; d < this.encHiddenDim; d++) {
        for (let a = 0; a < A; a++) {
          dEncStates[s][d] += this.Wa[a * this.encHiddenDim + d] * dCombined[a];
        }
      }

      // Through Wb * h_dec (same for all s, accumulate)
      for (let a = 0; a < A; a++) dWb_dec[a] += dCombined[a];
    }

    // Grad through Wb * h_dec
    accum(this.dWb, outer(dWb_dec, decHidden));
    const dDecHidden = zeros(this.decHiddenDim);
    for (let d = 0; d < this.decHiddenDim; d++) {
      for (let a = 0; a < A; a++) {
        dDecHidden[d] += this.Wb[a * this.decHiddenDim + d] * dWb_dec[a];
      }
    }

    return { dEncStates, dDecHidden };
  }

  zeroGrad() {
    this.dWa.fill(0);
    this.dWb.fill(0);
    this.dv.fill(0);
  }

  parameters() {
    return [
      { w: this.Wa, g: this.dWa },
      { w: this.Wb, g: this.dWb },
      { w: this.v,  g: this.dv  },
    ];
  }

  // ── Serialization ────────────────────────────────────────────────────────────

  save(filePath) {
    fs.writeFileSync(filePath, JSON.stringify({
      encHiddenDim: this.encHiddenDim,
      decHiddenDim: this.decHiddenDim,
      attnDim:      this.attnDim,
      Wa: Array.from(this.Wa),
      Wb: Array.from(this.Wb),
      v:  Array.from(this.v),
    }));
    console.log(`💾 Attention saved → ${filePath}`);
  }

  load(filePath) {
    const d = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    this.Wa = Float64Array.from(d.Wa);
    this.Wb = Float64Array.from(d.Wb);
    this.v  = Float64Array.from(d.v);
    console.log(`📂 Attention loaded ← ${filePath}`);
  }
}