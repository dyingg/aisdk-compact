import { generateText, wrapLanguageModel, gateway, type ModelMessage } from "ai";
import { compactMiddleware } from "aisdk-compact";

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
      content: `This is user message #${i + 1}. ` + "x".repeat(200),
    });
    messages.push({
      role: "assistant",
      content: `This is assistant response #${i + 1}. ` + "y".repeat(200),
    });
  }
  // End with a user message so the model has something to reply to
  messages.push({ role: "user", content: "Summarize our conversation so far." });
  return messages;
}

// ─── scenario 1: no compaction needed ───────────────────────────────────────

async function runShortConversation() {
  divider("Scenario 1: Short conversation (no compaction)");

  let compacted = false;

  const model = wrapLanguageModel({
    model: gateway("anthropic/claude-sonnet-4"),
    middleware: compactMiddleware({
      threshold: 0.8,
      onCompaction: (info) => {
        compacted = true;
        console.log("[compaction]", info);
      },
    }),
  });

  const messages = buildConversation(3); // 7 messages total — well under threshold
  console.log(`Sending ${messages.length} messages...`);

  const { text, usage } = await generateText({
    model,
    system: "You are a helpful assistant.",
    messages,
  });

  console.log(`Compaction triggered: ${compacted}`);
  console.log(`Usage: ${usage.inputTokens ?? "?"} input / ${usage.outputTokens ?? "?"} output tokens`);
  console.log(`Response:\n${text.slice(0, 300)}${text.length > 300 ? "..." : ""}`);
}

// ─── scenario 2: compaction triggered ───────────────────────────────────────

async function runLongConversation() {
  divider("Scenario 2: Long conversation (compaction triggered)");

  let compacted = false;

  const model = wrapLanguageModel({
    model: gateway("anthropic/claude-sonnet-4"),
    middleware: compactMiddleware({
      // Use a tiny maxTokens to force compaction with fewer messages
      maxTokens: 2_000,
      threshold: 0.5,
      recentMessageCount: 4,
      onCompaction: (info) => {
        compacted = true;
        console.log("[compaction]", info);
      },
    }),
  });

  const messages = buildConversation(20); // 41 messages — will exceed 2k * 0.5 = 1k token limit
  console.log(`Sending ${messages.length} messages...`);

  const { text, usage } = await generateText({
    model,
    system: "You are a helpful assistant.",
    messages,
  });

  console.log(`Compaction triggered: ${compacted}`);
  console.log(`Usage: ${usage.inputTokens ?? "?"} input / ${usage.outputTokens ?? "?"} output tokens`);
  console.log(`Response:\n${text.slice(0, 300)}${text.length > 300 ? "..." : ""}`);
}

// ─── run ────────────────────────────────────────────────────────────────────

async function main() {
  await runShortConversation();
  await runLongConversation();
}

main().catch(console.error);
