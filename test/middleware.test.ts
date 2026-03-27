import { describe, it, expect, vi } from "vitest";
import type { LanguageModelV3GenerateResult } from "@ai-sdk/provider";
import { MockLanguageModelV3 } from "ai/test";
import { compactMiddleware } from "../src/middleware.js";

function makeMockModel(modelId: string, summaryText = "Compacted summary.") {
  return new MockLanguageModelV3({
    modelId,
    doGenerate: {
      content: [{ type: "text", text: summaryText }],
      finishReason: { type: "stop" },
      usage: { inputTokens: 100, outputTokens: 50 },
      warnings: [],
    } as unknown as LanguageModelV3GenerateResult,
  });
}

function makeLargePrompt(messageCount: number) {
  const msgs = [{ role: "system" as const, content: "You are helpful." }];
  for (let i = 0; i < messageCount; i++) {
    msgs.push({
      role: "user" as const,
      content: [{ type: "text" as const, text: `Message ${i}: ${"x".repeat(500)}` }],
    } as any);
    msgs.push({
      role: "assistant" as const,
      content: [{ type: "text" as const, text: `Response ${i}: ${"y".repeat(500)}` }],
    } as any);
  }
  return msgs;
}

describe("compactMiddleware", () => {
  it("passes through when under threshold", async () => {
    const model = makeMockModel("gpt-4o");
    const middleware = compactMiddleware({
      maxTokens: 128_000,
      threshold: 0.8,
    });

    const params = {
      prompt: [
        { role: "system" as const, content: "Hi" },
        {
          role: "user" as const,
          content: [{ type: "text" as const, text: "Hello" }],
        },
      ],
    };

    const result = await middleware.transformParams!({
      type: "generate",
      params: params as any,
      model,
    });

    expect(result.prompt).toEqual(params.prompt);
  });

  it("compacts when over threshold", async () => {
    const model = makeMockModel("gpt-4o");
    const onCompaction = vi.fn();

    const middleware = compactMiddleware({
      maxTokens: 1000, // Very small to trigger compaction easily
      threshold: 0.1,
      recentMessageCount: 2,
      onCompaction,
    });

    const prompt = makeLargePrompt(10);

    const result = await middleware.transformParams!({
      type: "generate",
      params: { prompt } as any,
      model,
    });

    // Should have compacted: 1 system + 1 summary + 2 recent = 4
    expect(result.prompt.length).toBe(4);
    expect(onCompaction).toHaveBeenCalledOnce();
    expect(onCompaction.mock.calls[0][0].removedMessageCount).toBeGreaterThan(0);
  });

  it("auto-resolves maxTokens from model ID", async () => {
    const model = makeMockModel("gpt-4o-2024-08-06");
    const middleware = compactMiddleware({
      // No maxTokens — should auto-resolve to 128k for gpt-4o
      threshold: 0.8,
    });

    const params = {
      prompt: [
        {
          role: "user" as const,
          content: [{ type: "text" as const, text: "Short message" }],
        },
      ],
    };

    // Should not throw and should pass through (small message)
    const result = await middleware.transformParams!({
      type: "generate",
      params: params as any,
      model,
    });
    expect(result.prompt).toEqual(params.prompt);
  });

  it("passes through when model ID is unrecognized and no maxTokens", async () => {
    const model = makeMockModel("some-unknown-model-v42");
    const middleware = compactMiddleware({
      threshold: 0.8,
    });

    const prompt = makeLargePrompt(100);

    const result = await middleware.transformParams!({
      type: "generate",
      params: { prompt } as any,
      model,
    });

    // Should pass through unchanged since we can't determine context size
    expect(result.prompt).toEqual(prompt);
  });

  it("uses custom compaction model when provided", async () => {
    const wrappedModel = makeMockModel("gpt-4o");
    const compactionModel = makeMockModel("gpt-4o-mini", "Summary from mini model.");
    const compactionSpy = vi.spyOn(compactionModel, "doGenerate");

    const middleware = compactMiddleware({
      maxTokens: 500,
      threshold: 0.1,
      recentMessageCount: 2,
      compactionModel,
    });

    const prompt = makeLargePrompt(10);

    await middleware.transformParams!({
      type: "generate",
      params: { prompt } as any,
      model: wrappedModel,
    });

    // compactionModel should have been called, not the wrapped model
    expect(compactionSpy).toHaveBeenCalledOnce();
  });

  it("uses custom estimateTokens function", async () => {
    const model = makeMockModel("gpt-4o");
    const customEstimate = vi.fn().mockReturnValue(10); // Always under

    const middleware = compactMiddleware({
      maxTokens: 100,
      threshold: 0.8,
      estimateTokens: customEstimate,
    });

    const prompt = makeLargePrompt(50);

    await middleware.transformParams!({
      type: "generate",
      params: { prompt } as any,
      model,
    });

    expect(customEstimate).toHaveBeenCalled();
  });
});
