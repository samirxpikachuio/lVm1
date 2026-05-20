/**
 * setup.js
 * Run this ONCE before anything else.
 *
 * What it does:
 *   1. Checks Node.js version (need ≥ 18 for ES modules + structuredClone)
 *   2. Creates package.json  (sets "type":"module" so ES imports work)
 *   3. Validates dataset.json structure
 *   4. Prints a project summary and the exact commands to run next
 *
 * Usage:
 *   node setup.js
 */

import fs   from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── ANSI colours ─────────────────────────────────────────────────────────────
const C = {
  reset:  "\x1b[0m",
  bold:   "\x1b[1m",
  dim:    "\x1b[2m",
  green:  "\x1b[32m",
  red:    "\x1b[31m",
  yellow: "\x1b[33m",
  cyan:   "\x1b[36m",
  grey:   "\x1b[90m",
};
const ok   = (s) => `${C.green}${C.bold}  ✔${C.reset}  ${s}`;
const fail = (s) => `${C.red}${C.bold}  ✘${C.reset}  ${s}`;
const warn = (s) => `${C.yellow}${C.bold}  ⚠${C.reset}  ${s}`;
const info = (s) => `${C.grey}     ${s}${C.reset}`;

let errors = 0;

// ─── 1. Node version ──────────────────────────────────────────────────────────

function checkNode() {
  const [major, minor] = process.versions.node.split(".").map(Number);
  const version = process.versions.node;

  if (major < 18) {
    console.log(fail(`Node.js ${version} is too old. Need ≥ 18.0.0`));
    console.log(info("Download: https://nodejs.org"));
    errors++;
  } else {
    console.log(ok(`Node.js ${version}  ✓`));
  }
}

// ─── 2. package.json ──────────────────────────────────────────────────────────

function ensurePackageJson() {
  const pkgPath = path.join(__dirname, "package.json");

  if (fs.existsSync(pkgPath)) {
    const pkg = JSON.parse(fs.readFileSync(pkgPath, "utf-8"));
    if (pkg.type !== "module") {
      pkg.type = "module";
      fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
      console.log(warn(`package.json existed but was missing "type":"module" — fixed`));
    } else {
      console.log(ok(`package.json  →  "type":"module"  ✓`));
    }
    return;
  }

  const pkg = {
    name:    "chat-ai",
    version: "0.1.0",
    type:    "module",
    description: "From-scratch GRU seq2seq chat AI",
    scripts: {
      tokenize: "node tokenizer.js",
      embed:    "node embedding.js",
      train:    "node train.js",
      chat:     "node chat.js",
    },
  };

  fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2));
  console.log(ok(`package.json created  (type: module)`));
}

// ─── 3. Validate dataset.json ─────────────────────────────────────────────────

function validateDataset() {
  const dsPath = path.join(__dirname, "dataset.json");

  if (!fs.existsSync(dsPath)) {
    console.log(fail(`dataset.json not found in ${__dirname}`));
    console.log(info("Create it as an array of { \"question\": \"...\", \"answer\": \"...\" } objects."));
    errors++;
    return;
  }

  let dataset;
  try {
    dataset = JSON.parse(fs.readFileSync(dsPath, "utf-8"));
  } catch (e) {
    console.log(fail(`dataset.json is not valid JSON: ${e.message}`));
    errors++;
    return;
  }

  if (!Array.isArray(dataset)) {
    console.log(fail("dataset.json must be a JSON array"));
    errors++;
    return;
  }

  if (dataset.length === 0) {
    console.log(fail("dataset.json is empty"));
    errors++;
    return;
  }

  // Check every entry
  let bad = 0;
  for (let i = 0; i < dataset.length; i++) {
    const { question, answer } = dataset[i] ?? {};
    if (typeof question !== "string" || typeof answer !== "string") bad++;
  }

  if (bad > 0) {
    console.log(warn(`${bad} entries in dataset.json are missing "question" or "answer" fields`));
  }

  // Size guidance
  const pairs = dataset.length;
  let sizeNote;
  if (pairs < 50)        sizeNote = `${C.yellow}very small — model may overfit${C.reset}`;
  else if (pairs < 200)  sizeNote = `${C.yellow}small — workable, add more data over time${C.reset}`;
  else if (pairs < 1000) sizeNote = `${C.green}good size for GRU+attention${C.reset}`;
  else                   sizeNote = `${C.green}large — great${C.reset}`;

  console.log(ok(`dataset.json  →  ${pairs} Q&A pairs  (${sizeNote})`));

  // Avg lengths
  const avgQ = (dataset.reduce((s, d) => s + (d.question?.split(" ").length ?? 0), 0) / pairs).toFixed(1);
  const avgA = (dataset.reduce((s, d) => s + (d.answer?.split(" ").length ?? 0), 0) / pairs).toFixed(1);
  console.log(info(`avg question length: ${avgQ} words`));
  console.log(info(`avg answer length  : ${avgA} words`));
}

// ─── 4. Check all source files present ───────────────────────────────────────

function checkFiles() {
  const required = [
    "tokenizer.js",
    "embedding.js",
    "math.js",
    "encoder.js",
    "attention.js",
    "decoder.js",
    "model.js",
    "train.js",
    "chat.js",
  ];

  let allPresent = true;
  for (const f of required) {
    const fp = path.join(__dirname, f);
    if (!fs.existsSync(fp)) {
      console.log(fail(`Missing: ${f}`));
      allPresent = false;
      errors++;
    }
  }
  if (allPresent) console.log(ok(`All 9 source files present  ✓`));
}

// ─── 5. Print next-steps instructions ────────────────────────────────────────

function printInstructions() {
  console.log(`
${C.cyan}${C.bold}╔══════════════════════════════════════════════════════════╗
║              Chat AI — execution order                   ║
╚══════════════════════════════════════════════════════════╝${C.reset}

${C.bold}Step 1 — Tokenize${C.reset}
${C.green}  node tokenizer.js${C.reset}
${C.grey}  Reads dataset.json → builds vocab.json + encoded_dataset.json${C.reset}

${C.bold}Step 2 — Verify embeddings${C.reset}  ${C.dim}(optional but recommended)${C.reset}
${C.green}  node embedding.js${C.reset}
${C.grey}  Loads vocab → runs a demo forward/backward pass → saves embedding_weights.json${C.reset}

${C.bold}Step 3 — Train${C.reset}
${C.green}  node train.js${C.reset}
${C.grey}  Full training loop. Checkpoints saved to ./checkpoints/
  Best model auto-saved to ./checkpoints/best/
  Loss history saved to loss_history.json${C.reset}

${C.bold}Step 4 — Chat${C.reset}
${C.green}  node chat.js${C.reset}
${C.grey}  Interactive CLI. Type your message and press Enter.

  Flags:
    --greedy          deterministic output (best for debugging)
    --temp 0.6        lower = more focused  (default 0.8)
    --model ./checkpoints/epoch_30   load a specific checkpoint

  Commands inside the chat:
    mode     toggle greedy ↔ sampling
    reload   hot-reload the model weights without restarting
    exit     quit${C.reset}

${C.bold}npm shortcuts${C.reset}  ${C.dim}(after package.json is created)${C.reset}
${C.grey}  npm run tokenize
  npm run train
  npm run chat${C.reset}

${C.bold}Files produced during the pipeline:${C.reset}
${C.grey}  vocab.json              — word → id mapping (reusable)
  encoded_dataset.json    — integer-encoded + padded sequences
  embedding_weights.json  — initial embedding matrix (overwritten during train)
  loss_history.json       — per-epoch loss log
  checkpoints/
    best/                 — best weights seen during training
    epoch_10/             — checkpoint every 10 epochs
    epoch_20/
    ...${C.reset}
`);
}

// ─── Run ─────────────────────────────────────────────────────────────────────

console.log(`\n${C.cyan}${C.bold}  Chat AI — Setup Check${C.reset}\n`);

checkNode();
ensurePackageJson();
validateDataset();
checkFiles();

console.log();

if (errors > 0) {
  console.log(`${C.red}${C.bold}  ${errors} problem(s) found. Fix them before proceeding.${C.reset}\n`);
  process.exit(1);
} else {
  console.log(`${C.green}${C.bold}  All checks passed. You are ready to go!${C.reset}`);
  printInstructions();
}