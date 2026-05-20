/**
 * chat.js
 * Interactive CLI — talk to your trained ChatModel.
 *
 * Run:
 *   node chat.js
 *
 * Optional flags:
 *   --greedy          use argmax instead of sampling (more predictable)
 *   --temp 0.7        set sampling temperature (default 0.8)
 *   --model ./path    path to checkpoint directory (default: ./checkpoints/best)
 */

import fs            from "fs";
import path          from "path";
import readline      from "readline";
import { fileURLToPath } from "url";

import { Tokenizer, SPECIAL_TOKENS } from "./tokenizer.js";
import { ChatModel }                  from "./model.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ─── Parse CLI args ───────────────────────────────────────────────────────────

const args = process.argv.slice(2);
const getArg = (flag, def) => {
  const i = args.indexOf(flag);
  return i !== -1 && args[i + 1] ? args[i + 1] : def;
};

const GREEDY     = args.includes("--greedy");
const TEMP       = parseFloat(getArg("--temp", "0.8"));
const MODEL_DIR  = getArg("--model", path.join(__dirname, "checkpoints", "best"));
const VOCAB_PATH = path.join(__dirname, "vocab.json");

// ─── Colours (ANSI) ───────────────────────────────────────────────────────────

const C = {
  reset:  "\x1b[0m",
  bold:   "\x1b[1m",
  dim:    "\x1b[2m",
  cyan:   "\x1b[36m",
  green:  "\x1b[32m",
  yellow: "\x1b[33m",
  grey:   "\x1b[90m",
};

function colorize(text, ...codes) {
  return codes.join("") + text + C.reset;
}

// ─── Load model + tokenizer ───────────────────────────────────────────────────

function loadAssets() {
  if (!fs.existsSync(VOCAB_PATH)) {
    console.error(`${C.bold}❌ vocab.json not found.${C.reset} Run tokenizer.js first.`);
    process.exit(1);
  }
  if (!fs.existsSync(MODEL_DIR)) {
    console.error(`${C.bold}❌ Model directory not found:${C.reset} ${MODEL_DIR}`);
    console.error("   Run train.js first.");
    process.exit(1);
  }

  const tokenizer = new Tokenizer();
  tokenizer.loadVocab(VOCAB_PATH);

  const model = ChatModel.fromSaved(MODEL_DIR);
  model.cfg.temperature = TEMP;

  return { tokenizer, model };
}

// ─── Answer generation ────────────────────────────────────────────────────────

/**
 * Given a raw question string, return the model's answer string.
 */
function answer(question, tokenizer, model, greedy) {
  const qIds  = tokenizer.encode(question);
  const aIds  = model.generate(qIds, { greedy });
  return tokenizer.decode(aIds, { skipSpecial: true }) || "…";
}

// ─── Chat loop ────────────────────────────────────────────────────────────────

async function main() {
  console.log();
  console.log(colorize("  ╔══════════════════════════════════╗", C.cyan, C.bold));
  console.log(colorize("  ║        Chat AI  •  v0.1          ║", C.cyan, C.bold));
  console.log(colorize("  ╚══════════════════════════════════╝", C.cyan, C.bold));
  console.log();
  console.log(colorize(`  Model   : ${MODEL_DIR}`, C.grey));
  console.log(colorize(`  Mode    : ${GREEDY ? "greedy (argmax)" : `sampling  temp=${TEMP}`}`, C.grey));
  console.log(colorize(`  Commands: ${C.reset}${C.bold}exit${C.reset}${C.grey} to quit, ${C.reset}${C.bold}mode${C.reset}${C.grey} to toggle greedy/sample`, C.grey));
  console.log();

  // Load assets
  process.stdout.write(colorize("  Loading model…", C.dim));
  let { tokenizer, model } = loadAssets();
  process.stdout.write("\r" + " ".repeat(30) + "\r"); // clear line

  let greedy = GREEDY;

  const rl = readline.createInterface({
    input:  process.stdin,
    output: process.stdout,
  });

  const prompt = () => {
    rl.question(colorize("  You  › ", C.green, C.bold), async (input) => {
      input = input.trim();

      if (!input) { prompt(); return; }

      // Commands
      if (input.toLowerCase() === "exit" || input.toLowerCase() === "quit") {
        console.log(colorize("\n  Goodbye! 👋\n", C.cyan));
        rl.close();
        return;
      }

      if (input.toLowerCase() === "mode") {
        greedy = !greedy;
        console.log(colorize(`  ⚙  Mode switched to: ${greedy ? "greedy" : `sampling (temp=${TEMP})`}`, C.yellow));
        prompt();
        return;
      }

      if (input.toLowerCase() === "reload") {
        process.stdout.write(colorize("  Reloading model…", C.dim));
        ({ tokenizer, model } = loadAssets());
        process.stdout.write("\r" + " ".repeat(30) + "\r");
        console.log(colorize("  ✅ Model reloaded", C.yellow));
        prompt();
        return;
      }

      // Generate response
      try {
        const start = Date.now();
        const reply = answer(input, tokenizer, model, greedy);
        const ms    = Date.now() - start;

        console.log(
          colorize("  AI   › ", C.cyan, C.bold) +
          colorize(reply, C.reset) +
          colorize(`  (${ms}ms)`, C.grey)
        );
      } catch (err) {
        console.log(colorize(`  ⚠ Error: ${err.message}`, C.yellow));
      }

      console.log();
      prompt();
    });
  };

  prompt();
}

main();