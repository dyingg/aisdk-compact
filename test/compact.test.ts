import { describe, it, expect, vi } from "vitest";
import type { LanguageModelV3Prompt, LanguageModelV3GenerateResult } from "@ai-sdk/provider";
import { MockLanguageModelV3 } from "ai/test";
import { compactMessages } from "../src/compact.js";
import { ALICE_SENTENCES, ALICE_RESPONSES } from "./fixtures.js";

function makeMockModel(summaryText: string) {
  return new MockLanguageModelV3({
    doGenerate: {
      content: [{ type: "text", text: summaryText }],
      finishReason: { type: "stop" },
      usage: { inputTokens: 100, outputTokens: 50 },
      warnings: [],
    } as unknown as LanguageModelV3GenerateResult,
  });
}

function makeMessages(count: number): LanguageModelV3Prompt {
  const msgs: LanguageModelV3Prompt = [{ role: "system", content: "You are a helpful assistant." }];
  for (let i = 0; i < count; i++) {
    msgs.push({
      role: "user",
      content: [{ type: "text", text: ALICE_SENTENCES[i % ALICE_SENTENCES.length] }],
    });
    msgs.push({
      role: "assistant",
      content: [{ type: "text", text: ALICE_RESPONSES[i % ALICE_RESPONSES.length] }],
    });
  }
  return msgs;
}

describe("compactMessages", () => {
  it("returns prompt unchanged when fewer messages than recentMessageCount", async () => {
    const model = makeMockModel("summary");
    const prompt = makeMessages(2); // system + 4 conv messages = 5 total
    const result = await compactMessages(model, prompt, {
      recentMessageCount: 6,
    });
    expect(result).toEqual(prompt);
  });

  it("compacts older messages and preserves recent ones", async () => {
    const model = makeMockModel("This is the summary of older messages.");
    // 1 system + 20 conversation messages (10 pairs)
    const prompt = makeMessages(10);

    const result = await compactMessages(model, prompt, {
      recentMessageCount: 4,
    });

    // Should be: 1 system + 1 summary + 4 recent = 6 messages
    expect(result).toHaveLength(6);
    expect(result[0].role).toBe("system");
    expect(result[1].role).toBe("assistant");
    // Summary message should contain the compaction marker
    const summaryContent = result[1].content;
    expect(Array.isArray(summaryContent)).toBe(true);
    if (Array.isArray(summaryContent)) {
      const textPart = summaryContent[0];
      expect(textPart.type).toBe("text");
      if (textPart.type === "text") {
        expect(textPart.text).toContain("Compacted conversation summary");
        expect(textPart.text).toContain("originalMessages: 20");
        expect(textPart.text).toContain("messagesAfterCompaction: 5");
        expect(textPart.text).toContain("This is the summary of older messages.");
      }
    }

    // Last 4 messages should be preserved exactly
    expect(result.slice(2)).toEqual(prompt.slice(-4));
  });

  it("preserves system messages separately", async () => {
    const model = makeMockModel("summary");
    const prompt: LanguageModelV3Prompt = [
      { role: "system", content: "System prompt 1" },
      { role: "system", content: "System prompt 2" },
      ...Array.from({ length: 10 }, (_, i) => ({
        role: "user" as const,
        content: [{ type: "text" as const, text: `msg ${i}` }],
      })),
    ];

    const result = await compactMessages(model, prompt, {
      recentMessageCount: 2,
    });

    // 2 system + 1 summary + 2 recent = 5
    expect(result).toHaveLength(5);
    expect(result[0]).toEqual({
      role: "system",
      content: "System prompt 1",
    });
    expect(result[1]).toEqual({
      role: "system",
      content: "System prompt 2",
    });
    expect(result[2].role).toBe("assistant"); // summary
  });

  it("calls the model with serialized older messages", async () => {
    const model = makeMockModel("summary");
    const spy = vi.spyOn(model, "doGenerate");

    const prompt = makeMessages(5); // 1 system + 10 conv msgs

    await compactMessages(model, prompt, { recentMessageCount: 2 });

    expect(spy).toHaveBeenCalledOnce();
    const callArgs = spy.mock.calls[0][0];
    // The compaction prompt should be sent as a user message
    const userMsg = callArgs.prompt.find((m) => m.role === "user");
    expect(userMsg).toBeDefined();
    if (userMsg && userMsg.role === "user") {
      const text = userMsg.content[0];
      if (text.type === "text") {
        expect(text.text).toContain("tired of sitting by her sister");
        expect(text.text).toContain("conversation history to compact");
      }
    }
  });

  it("passes aisdk-compact providerOptions to doGenerate for trace identification", async () => {
    const model = makeMockModel("summary");
    const spy = vi.spyOn(model, "doGenerate");

    const prompt = makeMessages(5);

    await compactMessages(model, prompt, { recentMessageCount: 2 });

    const callArgs = spy.mock.calls[0][0];
    expect(callArgs.providerOptions).toEqual({
      "aisdk-compact": {
        purpose: "summarization",
        olderMessageCount: 8,
        recentMessageCount: 2,
      },
    });
    expect(callArgs.headers).toEqual({
      "X-Aisdk-Compact-Purpose": "summarization",
    });
  });

  it("does not produce bare console.log calls during compaction", async () => {
    const logSpy = vi.spyOn(console, "log").mockImplementation(() => {});
    const model = makeMockModel("summary");
    const prompt = makeMessages(5);

    await compactMessages(model, prompt, { recentMessageCount: 2 });

    expect(logSpy).not.toHaveBeenCalled();
    logSpy.mockRestore();
  });

  it("logs debug output when debug is true", async () => {
    const debugSpy = vi.spyOn(console, "debug").mockImplementation(() => {});
    const model = makeMockModel("summary");
    const prompt = makeMessages(5);

    await compactMessages(model, prompt, { recentMessageCount: 2, debug: true });

    const calls = debugSpy.mock.calls.map((c) => c[0]);
    expect(calls.some((c: string) => c.includes("[aisdk-compact] Summarizing"))).toBe(true);
    expect(calls.some((c: string) => c.includes("[aisdk-compact] Summary generated"))).toBe(true);
    debugSpy.mockRestore();
  });

  it("uses custom compaction prompt when provided", async () => {
    const model = makeMockModel("summary");
    const spy = vi.spyOn(model, "doGenerate");

    const prompt = makeMessages(5);
    const customPrompt = "Custom compaction instructions";

    await compactMessages(model, prompt, {
      recentMessageCount: 2,
      compactionPrompt: customPrompt,
    });

    const callArgs = spy.mock.calls[0][0];
    const systemMsg = callArgs.prompt.find((m) => m.role === "system");
    expect(systemMsg).toBeDefined();
    if (systemMsg && systemMsg.role === "system") {
      expect(systemMsg.content).toBe(customPrompt);
    }
  });
});
