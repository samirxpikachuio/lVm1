/**
 * encoder.js
 * GRU Encoder — reads a question token-by-token and produces:
 *   1. hiddenStates[t]  — hidden vector at every timestep  (used by Attention)
 *   2. finalHidden      — last hidden state                (seeds the Decoder)
 *
 * GRU equations (per timestep):
 *   z  = sigmoid(Wz·[h_prev, x] + bz)   ← update gate  (how much to update)
 *   r  = sigmoid(Wr·[h_prev, x] + br)   ← reset gate   (how much to forget)
 *   n  = tanh(Wn·[r⊙h_prev, x] + bn)   ← candidate    (new information)
 *   h  = (1-z)⊙h_prev + z⊙n            ← new hidden state
 *
 * Why GRU over LSTM?
 *   - Two gates instead of three → fewer params → better for small datasets
 *   - Empirically matches LSTM on short-to-medium sequences
 */

import fs   from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  zeros, vcopy, vadd, vmul, vscale,
  vsigmoid, vtanh, sigmoidGrad, tanhGrad,
  matvec, outer, accum,
  xavierUniform, matZeros,
} from "./math.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── GRUCell ──────────────────────────────────────────────────────────────────

/**
 * Single GRU timestep.
 * inputDim  = embedDim  (from EmbeddingLayer)
 * hiddenDim = user-chosen hidden size (e.g. 128)
 */
class GRUCell {
  constructor(inputDim, hiddenDim) {
    this.inputDim  = inputDim;
    this.hiddenDim = hiddenDim;
    const I = inputDim, H = hiddenDim;
    const concat = I + H; // size of [x, h_prev] concatenated

    // ── Weight matrices (row-major flat) ──────────────────────────────────────
    // Each gate: W shape [H × concat], b shape [H]
    this.Wz = xavierUniform(H * concat, concat, H);
    this.bz = zeros(H);

    this.Wr = xavierUniform(H * concat, concat, H);
    this.br = zeros(H);

    // Wn uses [x, r⊙h_prev] so same shape
    this.Wn = xavierUniform(H * concat, concat, H);
    this.bn = zeros(H);

    // ── Gradient accumulators (same shape as weights) ─────────────────────────
    this.dWz = matZeros(H, concat);
    this.dbz = zeros(H);
    this.dWr = matZeros(H, concat);
    this.dbr = zeros(H);
    this.dWn = matZeros(H, concat);
    this.dbn = zeros(H);

    // ── Cache for backprop ────────────────────────────────────────────────────
    this._cache = null;
  }

  // ── Forward ─────────────────────────────────────────────────────────────────

  /**
   * @param {Float64Array} x       input vector [inputDim]
   * @param {Float64Array} hPrev   previous hidden state [hiddenDim]
   * @returns {Float64Array}       new hidden state h [hiddenDim]
   */
  forward(x, hPrev) {
    const H = this.hiddenDim;

    // Concatenate [h_prev; x]  →  [H+I]
    const xh = new Float64Array([...hPrev, ...x]);

    // Update gate
    const zPre = vadd(matvec(this.Wz, xh, H, xh.length), this.bz);
    const z    = vsigmoid(zPre);

    // Reset gate
    const rPre = vadd(matvec(this.Wr, xh, H, xh.length), this.br);
    const r    = vsigmoid(rPre);

    // Candidate: use [x; r⊙h_prev]
    const rh   = new Float64Array([...vmul(r, hPrev), ...x]);
    const nPre = vadd(matvec(this.Wn, rh, H, rh.length), this.bn);
    const n    = vtanh(nPre);

    // New hidden state: h = (1-z)⊙h_prev + z⊙n
    const h = new Float64Array(H);
    for (let i = 0; i < H; i++) {
      h[i] = (1 - z[i]) * hPrev[i] + z[i] * n[i];
    }

    // Save everything needed for backward
    this._cache = { x, hPrev, xh, z, zPre, r, rPre, rh, n, nPre, h };
    return h;
  }

  // ── Backward ─────────────────────────────────────────────────────────────────

  /**
   * @param {Float64Array} dh   upstream gradient w.r.t. h [hiddenDim]
   * @returns {{ dInput: Float64Array, dHPrev: Float64Array }}
   */
  backward(dh) {
    const { x, hPrev, xh, z, r, rh, n, nPre, zPre, rPre } = this._cache;
    const H = this.hiddenDim, I = this.inputDim, C = xh.length;

    // Gradient through  h = (1-z)⊙h_prev + z⊙n
    const dn_raw = vmul(dh, z);                      // d_n = dh ⊙ z
    const dz_raw = new Float64Array(H);
    for (let i = 0; i < H; i++) {
      dz_raw[i] = dh[i] * (n[i] - hPrev[i]);        // d_z = dh ⊙ (n - h_prev)
    }
    const dhPrev_direct = new Float64Array(H);
    for (let i = 0; i < H; i++) {
      dhPrev_direct[i] = dh[i] * (1 - z[i]);        // from (1-z)⊙h_prev path
    }

    // Through tanh(nPre)
    const dnPre = vmul(dn_raw, tanhGrad(this._cache.n));
    accum(this.dWn, outer(dnPre, rh));
    accum(this.dbn, dnPre);
    const dRh = new Float64Array(rh.length);
    for (let i = 0; i < H; i++)
      for (let j = 0; j < rh.length; j++)
        dRh[j] += this.Wn[i * rh.length + j] * dnPre[i];

    // dRh = [d(r⊙h_prev) | d_x_from_n]
    const drh_hpart = dRh.slice(0, H);
    const dx_from_n = dRh.slice(H);

    // Through reset gate r
    const dr_raw  = vmul(drh_hpart, hPrev);          // d_r = d(r⊙h_prev) ⊙ h_prev
    const dhPrev_from_r = vmul(drh_hpart, r);        // d_h_prev from r path

    const drPre = vmul(dr_raw, sigmoidGrad(this._cache.r));
    accum(this.dWr, outer(drPre, xh));
    accum(this.dbr, drPre);
    const dxh_from_r = new Float64Array(C);
    for (let i = 0; i < H; i++)
      for (let j = 0; j < C; j++)
        dxh_from_r[j] += this.Wr[i * C + j] * drPre[i];

    // Through update gate z
    const dzPre = vmul(dz_raw, sigmoidGrad(this._cache.z));
    accum(this.dWz, outer(dzPre, xh));
    accum(this.dbz, dzPre);
    const dxh_from_z = new Float64Array(C);
    for (let i = 0; i < H; i++)
      for (let j = 0; j < C; j++)
        dxh_from_z[j] += this.Wz[i * C + j] * dzPre[i];

    // Sum all xh gradients
    const dxh = new Float64Array(C);
    for (let j = 0; j < C; j++) {
      dxh[j] = dxh_from_z[j] + dxh_from_r[j];
    }

    // Split dxh back into [dHPrev_part | dInput_part]
    const dHPrev = new Float64Array(H);
    const dInput = new Float64Array(I);
    for (let i = 0; i < H; i++) {
      dHPrev[i] = dxh[i] + dhPrev_direct[i] + dhPrev_from_r[i];
    }
    for (let i = 0; i < I; i++) {
      dInput[i] = dxh[H + i] + dx_from_n[i];
    }

    return { dInput, dHPrev };
  }

  zeroGrad() {
    this.dWz.fill(0); this.dbz.fill(0);
    this.dWr.fill(0); this.dbr.fill(0);
    this.dWn.fill(0); this.dbn.fill(0);
  }

  /** All trainable params and their grads — for the optimizer. */
  parameters() {
    return [
      { w: this.Wz, g: this.dWz }, { w: this.bz, g: this.dbz },
      { w: this.Wr, g: this.dWr }, { w: this.br, g: this.dbr },
      { w: this.Wn, g: this.dWn }, { w: this.bn, g: this.dbn },
    ];
  }
}

// ─── GRUEncoder ───────────────────────────────────────────────────────────────

/**
 * Runs a GRUCell over a full sequence of embeddings.
 * Returns all hidden states (for attention) + the final hidden state (for decoder seed).
 */
export class GRUEncoder {
  /**
   * @param {number} inputDim   — embedDim from EmbeddingLayer
   * @param {number} hiddenDim  — encoder hidden size (e.g. 128)
   */
  constructor(inputDim, hiddenDim) {
    this.inputDim  = inputDim;
    this.hiddenDim = hiddenDim;
    this.cell      = new GRUCell(inputDim, hiddenDim);

    // Cache for BPTT
    this._embeddings   = [];
    this._hiddenStates = []; // h[0] = initial zeros, h[t+1] = after step t
  }

  // ── Forward ─────────────────────────────────────────────────────────────────

  /**
   * @param {Float64Array[]} embeddings   [seqLen × inputDim]
   * @returns {{ hiddenStates: Float64Array[], finalHidden: Float64Array }}
   */
  forward(embeddings) {
    const seqLen = embeddings.length;
    this._embeddings = embeddings;

    let h = zeros(this.hiddenDim);
    const hiddenStates = [vcopy(h)]; // h_0 = zeros

    for (let t = 0; t < seqLen; t++) {
      h = this.cell.forward(embeddings[t], h);
      hiddenStates.push(vcopy(h));
    }

    this._hiddenStates = hiddenStates;
    return {
      hiddenStates: hiddenStates.slice(1), // h_1 … h_T  (exclude initial zero)
      finalHidden:  vcopy(h),
    };
  }

  // ── Backward (BPTT) ──────────────────────────────────────────────────────────

  /**
   * Backpropagation through time.
   *
   * @param {Float64Array[]} dHiddenStates  grad from attention for each h_t  [seqLen × hiddenDim]
   * @param {Float64Array}   dFinalHidden   grad from decoder for final h      [hiddenDim]
   * @returns {Float64Array[]}              grad w.r.t. each input embedding   [seqLen × inputDim]
   */
  backward(dHiddenStates, dFinalHidden) {
    const seqLen = this._embeddings.length;
    const dEmbeddings = [];

    // Carry gradient backward in time
    let dhNext = vcopy(dFinalHidden);

    for (let t = seqLen - 1; t >= 0; t--) {
      // Combine upstream attention grad + grad flowing from t+1
      const dhTotal = vadd(dHiddenStates[t], dhNext);

      // Re-run forward to restore cell cache for this timestep
      this.cell.forward(this._embeddings[t], this._hiddenStates[t]);
      const { dInput, dHPrev } = this.cell.backward(dhTotal);

      dEmbeddings[t] = dInput;
      dhNext = dHPrev;
    }

    return dEmbeddings;
  }

  zeroGrad() { this.cell.zeroGrad(); }

  parameters() { return this.cell.parameters(); }

  // ── Serialization ────────────────────────────────────────────────────────────

  save(filePath) {
    const data = {
      inputDim:  this.inputDim,
      hiddenDim: this.hiddenDim,
      cell: {
        Wz: Array.from(this.cell.Wz), bz: Array.from(this.cell.bz),
        Wr: Array.from(this.cell.Wr), br: Array.from(this.cell.br),
        Wn: Array.from(this.cell.Wn), bn: Array.from(this.cell.bn),
      },
    };
    fs.writeFileSync(filePath, JSON.stringify(data));
    console.log(`💾 Encoder saved → ${filePath}`);
  }

  load(filePath) {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    const c = data.cell;
    this.cell.Wz = Float64Array.from(c.Wz); this.cell.bz = Float64Array.from(c.bz);
    this.cell.Wr = Float64Array.from(c.Wr); this.cell.br = Float64Array.from(c.br);
    this.cell.Wn = Float64Array.from(c.Wn); this.cell.bn = Float64Array.from(c.bn);
    console.log(`📂 Encoder loaded ← ${filePath}`);
  }
}