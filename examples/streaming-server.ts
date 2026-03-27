import { Hono } from "hono";
import { streamText, wrapLanguageModel, gateway, type ModelMessage } from "ai";
import { compactMiddleware } from "aisdk-compact";

const app = new Hono();

const model = wrapLanguageModel({
  model: gateway("anthropic/claude-sonnet-4"),
  middleware: compactMiddleware({
    threshold: 0.8,
    recentMessageCount: 6,
    onCompaction: ({ originalTokens, compactedTokens, removedMessageCount }) => {
      console.log(
        `[compaction] ${removedMessageCount} messages removed | ${originalTokens} -> ${compactedTokens} tokens`
      );
    },
  }),
});

/**
 * POST /chat
 *
 * Body: { messages: ModelMessage[] }
 *
 * Returns a streaming text response compatible with the AI SDK's useChat hook.
 * Point any AI SDK frontend at http://localhost:3000/chat.
 */
app.post("/chat", async (c) => {
  const { messages } = await c.req.json<{ messages: ModelMessage[] }>();

  const result = streamText({
    model,
    system: "You are a helpful assistant. Keep responses concise.",
    messages,
  });

  return result.toTextStreamResponse();
});

app.get("/", (c) => {
  return c.text("aisdk-compact streaming example — POST /chat with { messages }");
});

console.log("Streaming server running on http://localhost:3000");

export default {
  port: 3000,
  fetch: app.fetch,
};
