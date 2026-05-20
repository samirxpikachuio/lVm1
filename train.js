/**
 * train.js
 * Training loop for ChatModel.
 *
 * Techniques used:
 *   Teacher forcing   — feed ground-truth answer tokens as decoder input
 *                       (faster, more stable training on small datasets)
 *   Shuffling         — randomise sample order each epoch to reduce overfitting
 *   Loss smoothing    — print rolling average loss (less noisy feedback)
 *   Checkpointing     — save best model weights automatically
 *   Early stopping    — halt if loss stops improving
 *
 * Run:
 *   node train.js
 */

import fs   from "fs";
import path from "path";
import { fileURLToPath } from "url";

import { Tokenizer, SPECIAL_TOKENS } from "./tokenizer.js";
import { ChatModel, DEFAULTS }        from "./model.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Training configuration ───────────────────────────────────────────────────

const CONFIG = {
  // Data
  datasetPath:  path.join(__dirname, "dataset.json"),
  vocabPath:    path.join(__dirname, "vocab.json"),
  encodedPath:  path.join(__dirname, "encoded_dataset.json"),
  checkpointDir: path.join(__dirname, "checkpoints"),

  // Training
  epochs:       60,    // number of full passes through the dataset
  logEvery:     5,     // print loss every N epochs
  saveEvery:    10,    // save checkpoint every N epochs
  earlyStop:    15,    // stop if best loss doesn't improve for N epochs

  // Model (override DEFAULTS here if needed)
  modelCfg: {
    embedDim:  64,
    hiddenDim: 128,
    attnDim:   64,
  },
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

/** Fisher-Yates shuffle (in-place). */
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
}

/** Rolling average helper. */
class RollingAvg {
  constructor(window = 50) { this.w = window; this.vals = []; }
  push(v) { this.vals.push(v); if (this.vals.length > this.w) this.vals.shift(); }
  get()   { return this.vals.reduce((a, b) => a + b, 0) / (this.vals.length || 1); }
}

/** Elapsed time as human-readable string. */
function elapsed(ms) {
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  return `${Math.floor(ms / 60000)}m ${((ms % 60000) / 1000).toFixed(0)}s`;
}

// ─── Main training routine ───────────────────────────────────────────────────

async function train() {
  // ── 1. Load tokenizer ──────────────────────────────────────────────────────
  if (!fs.existsSync(CONFIG.vocabPath)) {
    console.error("❌ vocab.json not found — run tokenizer.js first");
    process.exit(1);
  }
  const tokenizer = new Tokenizer();
  tokenizer.loadVocab(CONFIG.vocabPath);

  // ── 2. Load raw dataset ────────────────────────────────────────────────────
  const dataset = JSON.parse(fs.readFileSync(CONFIG.datasetPath, "utf-8"));
  console.log(`📖 Dataset: ${dataset.length} Q&A pairs`);

  // ── 3. Build training samples ──────────────────────────────────────────────
  // Each sample = { qIds, aIds } where:
  //   qIds = [<SOS>, <Q>, w1, w2, ..., <EOS>]
  //   aIds = [<A>,   w1, w2, ..., <EOS>]
  const samples = dataset.map(({ question, answer }) => ({
    qIds: tokenizer.encode(question),
    aIds: [
      SPECIAL_TOKENS["<A>"],
      ...tokenizer.encode(answer).slice(1), // slice off the <SOS> from encode()
    ],
  }));

  console.log(`🔢 Example qIds: [${samples[0].qIds.join(", ")}]`);
  console.log(`🔢 Example aIds: [${samples[0].aIds.join(", ")}]`);

  // ── 4. Build model ─────────────────────────────────────────────────────────
  const maxSeqLen = Math.max(...samples.map(s => Math.max(s.qIds.length, s.aIds.length))) + 5;
  const model = new ChatModel(tokenizer.vocabSize, maxSeqLen, CONFIG.modelCfg);

  if (!fs.existsSync(CONFIG.checkpointDir)) {
    fs.mkdirSync(CONFIG.checkpointDir, { recursive: true });
  }

  // ── 5. Training loop ───────────────────────────────────────────────────────
  let bestLoss      = Infinity;
  let noImprove     = 0;
  const rollingLoss = new RollingAvg(20);
  const lossHistory = [];
  const startTime   = Date.now();

  console.log(`\n${"─".repeat(60)}`);
  console.log(`🚀 Training for up to ${CONFIG.epochs} epochs`);
  console.log(`${"─".repeat(60)}\n`);

  for (let epoch = 1; epoch <= CONFIG.epochs; epoch++) {
    const order = samples.map((_, i) => i);
    shuffle(order);

    let epochLoss = 0;

    for (const idx of order) {
      const { qIds, aIds } = samples[idx];
      if (aIds.length < 2) continue; // skip degenerate samples

      const loss = model.trainStep(qIds, aIds);
      epochLoss += loss;
      rollingLoss.push(loss);
    }

    const avgLoss = epochLoss / order.length;
    lossHistory.push({ epoch, loss: +avgLoss.toFixed(6) });

    // ── Logging ───────────────────────────────────────────────────────────
    if (epoch % CONFIG.logEvery === 0 || epoch === 1) {
      const bar  = lossBar(avgLoss);
      const time = elapsed(Date.now() - startTime);
      console.log(
        `Epoch ${String(epoch).padStart(3)} │ loss ${avgLoss.toFixed(4)} │ ${bar} │ ${time}`
      );
    }

    // ── Checkpoint ────────────────────────────────────────────────────────
    if (epoch % CONFIG.saveEvery === 0) {
      const ckptDir = path.join(CONFIG.checkpointDir, `epoch_${epoch}`);
      model.save(ckptDir);
    }

    // ── Best model ────────────────────────────────────────────────────────
    if (avgLoss < bestLoss) {
      bestLoss  = avgLoss;
      noImprove = 0;
      model.save(path.join(CONFIG.checkpointDir, "best"));
    } else {
      noImprove++;
    }

    // ── Early stopping ────────────────────────────────────────────────────
    if (noImprove >= CONFIG.earlyStop) {
      console.log(`\n⏹  Early stopping — no improvement for ${CONFIG.earlyStop} epochs`);
      break;
    }
  }

  // ── 6. Final report ───────────────────────────────────────────────────────
  console.log(`\n${"─".repeat(60)}`);
  console.log(`✅ Training complete in ${elapsed(Date.now() - startTime)}`);
  console.log(`   Best loss : ${bestLoss.toFixed(6)}`);
  console.log(`   Best model: ${path.join(CONFIG.checkpointDir, "best")}/`);

  // Save loss history for plotting
  fs.writeFileSync(
    path.join(__dirname, "loss_history.json"),
    JSON.stringify(lossHistory, null, 2)
  );
  console.log(`   Loss log  : loss_history.json`);

  // ── 7. Quick sanity check on first 3 samples ──────────────────────────────
  console.log(`\n${"─".repeat(60)}`);
  console.log("🧪 Sanity check (greedy decode on training examples):\n");

  for (let i = 0; i < Math.min(3, samples.length); i++) {
    const { question, answer } = dataset[i];
    const genIds  = model.generate(samples[i].qIds, { greedy: true });
    const genText = tokenizer.decode(genIds, { skipSpecial: true });
    console.log(`  Q: ${question}`);
    console.log(`  A (expected) : ${answer}`);
    console.log(`  A (generated): ${genText}`);
    console.log();
  }
}

// ─── ASCII loss bar ───────────────────────────────────────────────────────────

function lossBar(loss, max = 5, width = 20) {
  const filled = Math.round((1 - Math.min(loss, max) / max) * width);
  return "█".repeat(filled) + "░".repeat(width - filled);
}

// ─── Entry point ─────────────────────────────────────────────────────────────

train().catch(err => {
  console.error("❌ Training failed:", err);
  process.exit(1);
});