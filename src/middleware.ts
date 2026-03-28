import type { LanguageModelV3Middleware } from "@ai-sdk/provider";
import type { CompactMiddlewareOptions } from "./types.js";
import { estimateTokens as defaultEstimateTokens } from "./token-estimator.js";
import { getModelContextSize } from "./model-context.js";
import { compactMessages } from "./compact.js";

/**
 * Creates a Vercel AI SDK middleware that automatically compacts conversation
 * context when token usage exceeds a threshold of the model's context window.
 */
export function compactMiddleware(
  options: CompactMiddlewareOptions = {}
): LanguageModelV3Middleware {
  const {
    maxTokens: configuredMaxTokens,
    threshold = 0.8,
    recentMessageCount = 6,
    compactionModel,
    compactionPrompt,
    estimateTokens = defaultEstimateTokens,
    onCompaction,
    debug = false,
  } = options;

  if (!compactionModel) {
    console.warn(
      "[aisdk-compact] No compactionModel provided. The base model will be used for compaction. " +
        "Consider using a cheaper/smaller model (e.g. gpt-4o-mini, claude-haiku) to reduce costs."
    );
  }

  if (debug) {
    console.debug(
      "[aisdk-compact] Middleware initialized",
      JSON.stringify({
        maxTokens: configuredMaxTokens ?? "auto",
        threshold,
        recentMessageCount,
        compactionModel: compactionModel ? "custom" : "base model",
      })
    );
  }

  return {
    specificationVersion: "v3",

    transformParams: async ({ params, model }) => {
      const messages = params.prompt;

      // Resolve max tokens: explicit config > model lookup
      const maxTokens = configuredMaxTokens ?? getModelContextSize(model.modelId);

      if (maxTokens == null) {
        console.warn(
          '[aisdk-compact] Could not auto-resolve context window for model "' +
            model.modelId +
            '". Defaulting to 60,000 tokens. Pass maxTokens to set the value explicitly.'
        );
      }

      const effectiveMaxTokens = maxTokens ?? 60_000;

      const tokenLimit = Math.floor(effectiveMaxTokens * threshold);
      const currentTokens = estimateTokens(messages);

      if (debug) {
        console.debug(
          `[aisdk-compact] Token estimate: ${currentTokens}/${effectiveMaxTokens} (threshold: ${tokenLimit}, ${((currentTokens / effectiveMaxTokens) * 100).toFixed(1)}% used)`
        );
      }

      if (currentTokens <= tokenLimit) {
        if (debug) {
          console.debug(
            `[aisdk-compact] Under threshold, passing through (${messages.length} messages)`
          );
        }
        return params;
      }

      if (debug) {
        console.debug("[aisdk-compact] Over threshold, triggering compaction...");
      }

      // Use the provided compaction model, or fall back to the wrapped model
      const summaryModel = compactionModel ?? model;

      const compactedPrompt = await compactMessages(summaryModel, messages, {
        recentMessageCount,
        compactionPrompt,
        debug,
      });

      if (onCompaction && compactedPrompt !== messages) {
        const compactedTokens = estimateTokens(compactedPrompt);
        // Count non-system messages that were in older portion
        const systemCount = messages.filter((m) => m.role === "system").length;
        const originalConvCount = messages.length - systemCount;
        const compactedConvCount =
          compactedPrompt.length - compactedPrompt.filter((m) => m.role === "system").length;
        const removedMessageCount = originalConvCount - compactedConvCount;

        onCompaction({
          originalTokens: currentTokens,
          compactedTokens,
          removedMessageCount,
        });
      }

      if (debug && compactedPrompt !== messages) {
        const compactedTokens = estimateTokens(compactedPrompt);
        console.debug(
          `[aisdk-compact] Compaction complete: ${messages.length} -> ${compactedPrompt.length} messages, ${currentTokens} -> ${compactedTokens} tokens`
        );
      }

      console.log("compactedPrompt", compactedPrompt);

      return { ...params, prompt: compactedPrompt };
    },
  };
}
