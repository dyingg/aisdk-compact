import { generateText, wrapLanguageModel, gateway } from "ai";
import type { ModelMessage } from "ai";
import { devToolsMiddleware } from "@ai-sdk/devtools";
import { compactMiddleware } from "../src/index.js";
import { ALICE_SENTENCES, ALICE_RESPONSES } from "../test/fixtures.js";

// ─── helpers ────────────────────────────────────────────────────────────────

function divider(title: string) {
  console.log(`\n${"=".repeat(70)}`);
  console.log(` ${title}`);
  console.log(`${"=".repeat(70)}\n`);
}

function buildConversation(turns: number): ModelMessage[] {
  const messages: ModelMessage[] = [];
  for (let i = 0; i < turns; i++) {
    messages.push({
      role: "user",
      content: ALICE_SENTENCES[i % ALICE_SENTENCES.length],
    });
    messages.push({
      role: "assistant",
      content: ALICE_RESPONSES[i % ALICE_RESPONSES.length],
    });
  }
  // End with a user message so the model has something to reply to
  messages.push({
    role: "user",
    content: "Count the number of tokens, and words in the conversation so far.",
  });
  return messages;
}

// ─── scenario 1: no compaction needed ───────────────────────────────────────

async function runShortConversation() {
  divider("Scenario 1: Short conversation (no compaction)");

  let compacted = false;

  const model = wrapLanguageModel({
    model: gateway("openai/gpt-4.1-nano"),
    middleware: [
      devToolsMiddleware(),
      compactMiddleware({
        threshold: 0.8,
        onCompaction: (info) => {
          compacted = true;
          console.log("[compaction]", info);
        },
      }),
    ],
  });

  const messages = buildConversation(3); // 7 messages total — well under threshold
  console.log(`Sending ${messages.length} messages...`);

  const { text, usage } = await generateText({
    model,
    system: "You are a helpful assistant.",
    messages,
  });

  console.log(`Compaction triggered: ${compacted}`);
  console.log(
    `Usage: ${usage.inputTokens ?? "?"} input / ${usage.outputTokens ?? "?"} output tokens`
  );
  console.log(`Response:\n${text.slice(0, 300)}${text.length > 300 ? "..." : ""}`);
}

// ─── scenario 2: compaction triggered ───────────────────────────────────────

async function runLongConversation() {
  divider("Scenario 2: Long conversation (compaction triggered)");

  let compacted = false;

  const model = wrapLanguageModel({
    model: gateway("openai/gpt-4.1-nano"),
    middleware: [
      compactMiddleware({
        maxTokens: 2_000,
        threshold: 0.5,
        recentMessageCount: 4,
        debug: true,
        onCompaction: (info) => {
          compacted = true;
          console.log("[compaction]", info);
        },
      }),
      devToolsMiddleware(),
    ],
  });

  const messages = buildConversation(30); // 61 messages — will exceed 2k * 0.5 = 1k token limit
  console.log(`Sending ${messages.length} messages...`);

  const { text, usage } = await generateText({
    model,
    system: "You are a helpful assistant.",
    messages,
  });
}

// ─── run ────────────────────────────────────────────────────────────────────

async function main() {
  // await runShortConversation();
  await runLongConversation();
}

main().catch(console.error);
