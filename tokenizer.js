/**
 * tokenizer.js
 * A from-scratch tokenizer for a chat AI trained on dataset.json
 *
 * Pipeline:
 *   raw text → normalize → split → build vocab → encode / decode
 *
 * Vocab structure:
 *   Special tokens  : <PAD>=0  <UNK>=1  <SOS>=2  <EOS>=3  <Q>=4  <A>=5
 *   Word tokens     : index 6 … N
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

// ─── Constants ────────────────────────────────────────────────────────────────

const SPECIAL_TOKENS = {
  "<PAD>": 0, // padding (align batches)
  "<UNK>": 1, // unknown word
  "<SOS>": 2, // start of sequence
  "<EOS>": 3, // end of sequence
  "<Q>":   4, // marks a question turn
  "<A>":   5, // marks an answer turn
};

const SPECIAL_TOKEN_COUNT = Object.keys(SPECIAL_TOKENS).length; // 6

// ─── Normalizer ───────────────────────────────────────────────────────────────

/**
 * Normalizes a raw string:
 *  - lowercase
 *  - expand common contractions
 *  - pad punctuation with spaces so they become separate tokens
 *  - collapse whitespace
 */
function normalize(text) {
  const contractions = {
    "can't": "can not", "won't": "will not", "n't": " not",
    "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
  };

  let t = text.toLowerCase().trim();

  for (const [k, v] of Object.entries(contractions)) {
    t = t.replaceAll(k, v);
  }

  // pad punctuation: .,!?;:()[]"'
  t = t.replace(/([.,!?;:()\[\]"'])/g, " $1 ");

  // collapse multiple spaces
  t = t.replace(/\s+/g, " ").trim();

  return t;
}

// ─── Tokenizer (word-level) ────────────────────────────────────────────────────

class Tokenizer {
  constructor() {
    /** word → id */
    this.vocab = { ...SPECIAL_TOKENS };
    /** id → word */
    this.reverseVocab = Object.fromEntries(
      Object.entries(SPECIAL_TOKENS).map(([k, v]) => [v, k])
    );
    this.nextId = SPECIAL_TOKEN_COUNT;
  }

  // ── Vocab building ──────────────────────────────────────────────────────────

  /** Add a single word to the vocab (no-op if already present). */
  addWord(word) {
    if (this.vocab[word] === undefined) {
      this.vocab[word] = this.nextId;
      this.reverseVocab[this.nextId] = word;
      this.nextId++;
    }
  }

  /**
   * Build vocabulary from the full dataset.
   * @param {Array<{question: string, answer: string}>} dataset
   */
  buildVocab(dataset) {
    for (const { question, answer } of dataset) {
      for (const text of [question, answer]) {
        const words = normalize(text).split(" ");
        for (const w of words) if (w) this.addWord(w);
      }
    }
    console.log(`✅ Vocab built — ${this.vocabSize} tokens (${SPECIAL_TOKEN_COUNT} special + ${this.vocabSize - SPECIAL_TOKEN_COUNT} words)`);
  }

  get vocabSize() {
    return this.nextId;
  }

  // ── Encoding ────────────────────────────────────────────────────────────────

  /**
   * Encode a single sentence to an array of token ids.
   * Wraps the sequence with <SOS> … <EOS>.
   * Unknown words map to <UNK>.
   *
   * @param {string} text
   * @returns {number[]}
   */
  encode(text) {
    const words = normalize(text).split(" ").filter(Boolean);
    const ids = [SPECIAL_TOKENS["<SOS>"]];
    for (const w of words) {
      ids.push(this.vocab[w] ?? SPECIAL_TOKENS["<UNK>"]);
    }
    ids.push(SPECIAL_TOKENS["<EOS>"]);
    return ids;
  }

  /**
   * Encode a question+answer pair as a single sequence:
   *   <SOS> <Q> [question tokens] <A> [answer tokens] <EOS>
   *
   * This single-sequence format is convenient for seq2seq / decoder-only models.
   *
   * @param {string} question
   * @param {string} answer
   * @returns {number[]}
   */
  encodePair(question, answer) {
    const qWords = normalize(question).split(" ").filter(Boolean);
    const aWords = normalize(answer).split(" ").filter(Boolean);

    return [
      SPECIAL_TOKENS["<SOS>"],
      SPECIAL_TOKENS["<Q>"],
      ...qWords.map(w => this.vocab[w] ?? SPECIAL_TOKENS["<UNK>"]),
      SPECIAL_TOKENS["<A>"],
      ...aWords.map(w => this.vocab[w] ?? SPECIAL_TOKENS["<UNK>"]),
      SPECIAL_TOKENS["<EOS>"],
    ];
  }

  // ── Decoding ────────────────────────────────────────────────────────────────

  /**
   * Decode an array of token ids back to a string.
   * Special tokens are shown in angle brackets so you can see the structure.
   *
   * @param {number[]} ids
   * @param {object}   [opts]
   * @param {boolean}  [opts.skipSpecial=false]  strip special tokens from output
   * @returns {string}
   */
  decode(ids, { skipSpecial = false } = {}) {
    const specialSet = new Set(Object.values(SPECIAL_TOKENS));
    const words = [];
    for (const id of ids) {
      if (skipSpecial && specialSet.has(id)) continue;
      words.push(this.reverseVocab[id] ?? "<UNK>");
    }
    return words.join(" ").trim();
  }

  // ── Padding / batching ──────────────────────────────────────────────────────

  /**
   * Pad or truncate a sequence to exactly `length` tokens.
   * Padding uses token id 0 (<PAD>).
   *
   * @param {number[]} ids
   * @param {number}   length
   * @returns {number[]}
   */
  padOrTruncate(ids, length) {
    if (ids.length >= length) return ids.slice(0, length);
    return [...ids, ...Array(length - ids.length).fill(SPECIAL_TOKENS["<PAD>"])];
  }

  /**
   * Batch-encode all pairs and pad them to the same length.
   *
   * @param {Array<{question:string, answer:string}>} dataset
   * @param {number} [maxLen]  explicit max length; defaults to longest sequence
   * @returns {{ encoded: number[][], maxLen: number }}
   */
  batchEncode(dataset, maxLen) {
    const sequences = dataset.map(({ question, answer }) =>
      this.encodePair(question, answer)
    );

    const longest = Math.max(...sequences.map(s => s.length));
    const targetLen = maxLen ?? longest;

    const encoded = sequences.map(s => this.padOrTruncate(s, targetLen));
    return { encoded, maxLen: targetLen };
  }

  // ── Serialization ───────────────────────────────────────────────────────────

  /** Save vocab to JSON so it can be reloaded without re-training. */
  saveVocab(filePath) {
    const data = {
      vocab: this.vocab,
      reverseVocab: this.reverseVocab,
      nextId: this.nextId,
      specialTokens: SPECIAL_TOKENS,
    };
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2));
    console.log(`💾 Vocab saved → ${filePath}`);
  }

  /** Load a previously saved vocab. */
  loadVocab(filePath) {
    const data = JSON.parse(fs.readFileSync(filePath, "utf-8"));
    this.vocab        = data.vocab;
    this.reverseVocab = data.reverseVocab;
    this.nextId       = data.nextId;
    console.log(`📂 Vocab loaded — ${this.vocabSize} tokens`);
  }
}

// ─── Main: run when executed directly ─────────────────────────────────────────

const __dirname = path.dirname(fileURLToPath(import.meta.url));

function main() {
  // 1. Load dataset
  const datasetPath = path.join(__dirname, "dataset.json");
  if (!fs.existsSync(datasetPath)) {
    console.error("❌ dataset.json not found next to tokenizer.js");
    process.exit(1);
  }
  const dataset = JSON.parse(fs.readFileSync(datasetPath, "utf-8"));
  console.log(`📖 Loaded ${dataset.length} Q&A pairs`);

  // 2. Build tokenizer & vocab
  const tokenizer = new Tokenizer();
  tokenizer.buildVocab(dataset);

  // 3. Save vocab for later reuse
  tokenizer.saveVocab(path.join(__dirname, "vocab.json"));

  // 4. Batch-encode full dataset
  const { encoded, maxLen } = tokenizer.batchEncode(dataset);
  console.log(`\n✅ Batch encoded ${encoded.length} sequences, padded to length ${maxLen}`);

  // 6. Save encoded dataset
  const outPath = path.join(__dirname, "encoded_dataset.json");
  fs.writeFileSync(outPath, JSON.stringify({ maxLen, encoded }, null, 2));
  console.log(`💾 Encoded dataset saved → ${outPath}`);
}

main();

// ─── Exports (use as a module in your training pipeline) ──────────────────────

export { Tokenizer, normalize, SPECIAL_TOKENS };