/**
 * decoder.js
 * GRU Decoder — generates the answer one token at a time.
 *
 * At each step t the decoder receives:
 *   - The previous token's embedding    (teacher-forced during training)
 *   - The context vector from Attention (what the encoder focused on)
 *   - Its own previous hidden state
 *
 * GRU input at each step = concat(prev_embedding, context_vector)
 * Output projection: h_t → logits over vocab → softmax → next token
 *
 *   input_t = [embed(y_{t-1}); context_t]
 *   h_t     = GRUCell(input_t, h_{t-1})
 *   logits  = Wo · h_t + bo
 *   probs   = softmax(logits)
 */

import fs from "fs";
import {
  zeros, vcopy, vadd, vscale,
  vsigmoid, vtanh, sigmoidGrad, tanhGrad,
  matvec, outer, accum,
  xavierUniform, matZeros,
  softmax, crossEntropy, crossEntropyGrad, vmul,
} from "./math.js";

// ─── GRUCell (decoder version) ────────────────────────────────────────────────
// Same math as encoder cell, just re-instantiated with different dims.
// inputDim = embedDim + encHiddenDim  (embedding + context vector concatenated)

class GRUCell {
  constructor(inputDim, hiddenDim) {
    this.inputDim  = inputDim;
    this.hiddenDim = hiddenDim;
    const C = inputDim + hiddenDim; // concat size

    this.Wz = xavierUniform(hiddenDim * C, C, hiddenDim);
    this.bz = zeros(hiddenDim);
    this.Wr = xavierUniform(hiddenDim * C, C, hiddenDim);
    this.br = zeros(hiddenDim);
    this.Wn = xavierUniform(hiddenDim * C, C, hiddenDim);
    this.bn = zeros(hiddenDim);

    this.dWz = matZeros(hiddenDim, C); this.dbz = zeros(hiddenDim);
    this.dWr = matZeros(hiddenDim, C); this.dbr = zeros(hiddenDim);
    this.dWn = matZeros(hiddenDim, C); this.dbn = zeros(hiddenDim);

    this._cache = null;
  }

  forward(x, hPrev) {
    const H = this.hiddenDim;
    const xh = new Float64Array([...hPrev, ...x]);
    const C  = xh.length;

    const zPre = vadd(matvec(this.Wz, xh, H, C), this.bz);
    const z    = vsigmoid(zPre);

    const rPre = vadd(matvec(this.Wr, xh, H, C), this.br);
    const r    = vsigmoid(rPre);

    const rh   = new Float64Array([...vmul(r, hPrev), ...x]);
    const nPre = vadd(matvec(this.Wn, rh, H, rh.length), this.bn);
    const n    = vtanh(nPre);

    const h = new Float64Array(H);
    for (let i = 0; i < H; i++) h[i] = (1 - z[i]) * hPrev[i] + z[i] * n[i];

    this._cache = { x, hPrev, xh, z, zPre, r, rPre, rh, n, nPre, h };
    return h;
  }

  backward(dh) {
    const { x, hPrev, xh, z, r, rh, n, nPre, zPre, rPre } = this._cache;
    const H = this.hiddenDim, I = this.inputDim, C = xh.length;

    const dn_raw = vmul(dh, z);
    const dz_raw = new Float64Array(H);
    for (let i = 0; i < H; i++) dz_raw[i] = dh[i] * (n[i] - hPrev[i]);
    const dhPrev_direct = new Float64Array(H);
    for (let i = 0; i < H; i++) dhPrev_direct[i] = dh[i] * (1 - z[i]);

    const dnPre = vmul(dn_raw, tanhGrad(this._cache.n));
    accum(this.dWn, outer(dnPre, rh));
    accum(this.dbn, dnPre);
    const dRh = new Float64Array(rh.length);
    for (let i = 0; i < H; i++)
      for (let j = 0; j < rh.length; j++)
        dRh[j] += this.Wn[i * rh.length + j] * dnPre[i];

    const drh_hpart  = dRh.slice(0, H);
    const dx_from_n  = dRh.slice(H);
    const dr_raw     = vmul(drh_hpart, hPrev);
    const dhPrev_from_r = vmul(drh_hpart, r);

    const drPre = vmul(dr_raw, sigmoidGrad(this._cache.r));
    accum(this.dWr, outer(drPre, xh));
    accum(this.dbr, drPre);
    const dxh_from_r = new Float64Array(C);
    for (let i = 0; i < H; i++)
      for (let j = 0; j < C; j++)
        dxh_from_r[j] += this.Wr[i * C + j] * drPre[i];

    const dzPre2 = vmul(dz_raw, sigmoidGrad(this._cache.z));
    accum(this.dWz, outer(dzPre2, xh));
    accum(this.dbz, dzPre2);
    const dxh_from_z = new Float64Array(C);
    for (let i = 0; i < H; i++)
      for (let j = 0; j < C; j++)
        dxh_from_z[j] += this.Wz[i * C + j] * dzPre2[i];

    const dxh = new Float64Array(C);
    for (let j = 0; j < C; j++) dxh[j] = dxh_from_z[j] + dxh_from_r[j];

    const dHPrev = new Float64Array(H);
    const dInput = new Float64Array(I);
    for (let i = 0; i < H; i++) dHPrev[i] = dxh[i] + dhPrev_direct[i] + dhPrev_from_r[i];
    for (let i = 0; i < I; i++) dInput[i]  = dxh[H + i] + dx_from_n[i];

    return { dInput, dHPrev };
  }

  zeroGrad() {
    this.dWz.fill(0); this.dbz.fill(0);
    this.dWr.fill(0); this.dbr.fill(0);
    this.dWn.fill(0); this.dbn.fill(0);
  }

  parameters() {
    return [
      { w: this.Wz, g: this.dWz }, { w: this.bz, g: this.dbz },
      { w: this.Wr, g: this.dWr }, { w: this.br, g: this.dbr },
      { w: this.Wn, g: this.dWn }, { w: this.bn, g: this.dbn },
    ];
  }
}

// ─── GRUDecoder ───────────────────────────────────────────────────────────────

export class GRUDecoder {
  /**
   * @param {number} embedDim       — word embedding dimension
   * @param {number} encHiddenDim   — encoder hidden size (context vector size)
   * @param {number} decHiddenDim   — decoder hidden size
   * @param {number} vocabSize      — output vocabulary size
   */
  constructor(embedDim, encHiddenDim, decHiddenDim, vocabSize) {
    this.embedDim     = embedDim;
    this.encHiddenDim = encHiddenDim;
    this.decHiddenDim = decHiddenDim;
    this.vocabSize    = vocabSize;

    // GRU cell: input = [embedding ‖ context]
    this.cell = new GRUCell(embedDim + encHiddenDim, decHiddenDim);

    // Output projection: h → logits over vocab
    this.Wo = xavierUniform(vocabSize * decHiddenDim, decHiddenDim, vocabSize);
    this.bo = zeros(vocabSize);
    this.dWo = matZeros(vocabSize, decHiddenDim);
    this.dbo = zeros(vocabSize);

    // Cache for backward
    this._steps = []; // one entry per decoding step
  }

  // ── Single step ──────────────────────────────────────────────────────────────

  /**
   * One decoding step.
   * @param {Float64Array} embedding   embedding of previous token [embedDim]
   * @param {Float64Array} context     attention context vector [encHiddenDim]
   * @param {Float64Array} hPrev       decoder hidden state [decHiddenDim]
   * @returns {{ h: Float64Array, logits: Float64Array, probs: Float64Array }}
   */
  step(embedding, context, hPrev) {
    // Concatenate embedding + context as GRU input
    const input = new Float64Array([...embedding, ...context]);
    const h     = this.cell.forward(input, hPrev);

    // Project to vocab
    const logits = vadd(matvec(this.Wo, h, this.vocabSize, this.decHiddenDim), this.bo);
    const probs  = softmax(Array.from(logits));

    this._steps.push({ embedding, context, hPrev, input, h, logits, probs });
    return { h, logits: Float64Array.from(logits), probs: Float64Array.from(probs) };
  }

  /** Clear step cache (call before each new sequence). */
  resetSteps() { this._steps = []; }

  // ── Backward ─────────────────────────────────────────────────────────────────

  /**
   * Backprop through all decoder steps.
   *
   * @param {number[]}       targetIds     ground-truth token ids [decSeqLen]
   * @returns {{
   *   loss:          number,
   *   dContexts:     Float64Array[],   // grad w.r.t. each context vector
   *   dEmbeddings:   Float64Array[],   // grad w.r.t. each input embedding
   *   dInitHidden:   Float64Array,     // grad w.r.t. initial hidden (= encoder final)
   * }}
   */
  backward(targetIds) {
    const T = this._steps.length;
    let totalLoss = 0;
    const dContexts   = [];
    const dEmbeddings = [];
    let dhNext = zeros(this.decHiddenDim);

    for (let t = T - 1; t >= 0; t--) {
      const { embedding, context, hPrev, input, h, probs } = this._steps[t];

      // Loss at this step
      totalLoss += crossEntropy(probs, targetIds[t]);

      // Grad of cross-entropy + softmax combined
      const dLogits = crossEntropyGrad(probs, targetIds[t]);

      // Grad through output projection
      accum(this.dWo, outer(dLogits, h));
      accum(this.dbo, dLogits);
      const dh = new Float64Array(this.decHiddenDim);
      for (let d = 0; d < this.decHiddenDim; d++)
        for (let v = 0; v < this.vocabSize; v++)
          dh[d] += this.Wo[v * this.decHiddenDim + d] * dLogits[v];

      // Add grad flowing from t+1
      const dhTotal = vadd(dh, dhNext);

      // Backprop through GRU cell
      this.cell.forward(input, hPrev); // restore cache
      const { dInput, dHPrev } = this.cell.backward(dhTotal);

      // Split dInput back into [dEmbedding | dContext]
      dEmbeddings[t] = dInput.slice(0, this.embedDim);
      dContexts[t]   = dInput.slice(this.embedDim);

      dhNext = dHPrev;
    }

    return {
      loss:        totalLoss / T,
      dContexts,
      dEmbeddings,
      dInitHidden: dhNext, // flows back to encoder final hidden
    };
  }

  zeroGrad() {
    this.cell.zeroGrad();
    this.dWo.fill(0);
    this.dbo.fill(0);
  }

  parameters() {
    return [
      ...this.cell.parameters(),
      { w: this.Wo, g: this.dWo },
      { w: this.bo, g: this.dbo },
    ];
  }

  // ── Serialization ────────────────────────────────────────────────────────────

  save(filePath) {
    const c = this.cell;
    fs.writeFileSync(filePath, JSON.stringify({
      embedDim: this.embedDim, encHiddenDim: this.encHiddenDim,
      decHiddenDim: this.decHiddenDim, vocabSize: this.vocabSize,
      cell: {
        Wz: Array.from(c.Wz), bz: Array.from(c.bz),
        Wr: Array.from(c.Wr), br: Array.from(c.br),
        Wn: Array.from(c.Wn), bn: Array.from(c.bn),
      },
      Wo: Array.from(this.Wo),
      bo: Array.from(this.bo),
    }));
    console.log(`💾 Decoder saved → ${filePath}`);
  }

  load(filePath) {
    const d = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    const c = d.cell;
    this.cell.Wz = Float64Array.from(c.Wz); this.cell.bz = Float64Array.from(c.bz);
    this.cell.Wr = Float64Array.from(c.Wr); this.cell.br = Float64Array.from(c.br);
    this.cell.Wn = Float64Array.from(c.Wn); this.cell.bn = Float64Array.from(c.bn);
    this.Wo = Float64Array.from(d.Wo);
    this.bo = Float64Array.from(d.bo);
    console.log(`📂 Decoder loaded ← ${filePath}`);
  }
}