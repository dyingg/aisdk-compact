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

function pad(value: number, width = 15): string {
  return String(value).padStart(width);
}

// ─── comparison ─────────────────────────────────────────────────────────────

async function runComparison() {
  const TURNS = 30;
  const messages = buildConversation(TURNS);

  // ── Run 1: Without compaction ──────────────────────────────────────────

  divider("Run 1: Without compaction");
  console.log(`Sending ${messages.length} messages...`);

  const plainModel = wrapLanguageModel({
    model: gateway("openai/gpt-4.1-nano"),
    middleware: [devToolsMiddleware()],
  });

  const plainResult = await generateText({
    model: plainModel,
    system: "You are a helpful assistant.",
    messages,
  });

  const plainInput = plainResult.usage.inputTokens;
  const plainOutput = plainResult.usage.outputTokens;

  console.log(`Input tokens:  ${plainInput}`);
  console.log(`Output tokens: ${plainOutput}`);

  // ── Run 2: With compaction ─────────────────────────────────────────────

  divider("Run 2: With compaction");
  console.log(`Sending ${messages.length} messages...`);

  const compactModel = wrapLanguageModel({
    model: gateway("openai/gpt-4.1-nano"),
    middleware: [
      devToolsMiddleware(),
      compactMiddleware({
        maxTokens: 2_000,
        threshold: 0.5,
        recentMessageCount: 4,
        onCompaction: (info) => {
          console.log(
            `[compaction] ${info.removedMessageCount} messages removed, ` +
              `${info.originalTokens} → ${info.compactedTokens} estimated tokens`
          );
        },
      }),
    ],
  });

  const compactResult = await generateText({
    model: compactModel,
    system: "You are a helpful assistant.",
    messages,
  });

  const compactInput = compactResult.usage.inputTokens;
  const compactOutput = compactResult.usage.outputTokens;

  console.log(`Input tokens:  ${compactInput}`);
  console.log(`Output tokens: ${compactOutput}`);

  // ── Comparison table ───────────────────────────────────────────────────

  divider("Comparison");

  console.log(
    `Metric          | Without Compact | With Compact`
  );
  console.log(
    `----------------|-----------------|-------------`
  );
  console.log(
    `Input tokens    | ${pad(plainInput)} | ${pad(compactInput)}`
  );
  console.log(
    `Output tokens   | ${pad(plainOutput)} | ${pad(compactOutput)}`
  );
  console.log(
    `Total tokens    | ${pad(plainInput + plainOutput)} | ${pad(compactInput + compactOutput)}`
  );
  console.log();

  const saved = plainInput - compactInput;
  const pct = plainInput > 0 ? ((saved / plainInput) * 100).toFixed(1) : "0";
  console.log(`Savings: ${saved} input tokens saved (${pct}%)`);
}

// ─── run ────────────────────────────────────────────────────────────────────

runComparison().catch(console.error);
