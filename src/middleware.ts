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
  } = options;

  if (!compactionModel) {
    console.warn(
      "[aisdk-compact] No compactionModel provided. The base model will be used for compaction. " +
        "Consider using a cheaper/smaller model (e.g. gpt-4o-mini, claude-haiku) to reduce costs."
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

      if (currentTokens <= tokenLimit) {
        return params;
      }

      // Use the provided compaction model, or fall back to the wrapped model
      const summaryModel = compactionModel ?? model;

      const compactedPrompt = await compactMessages(summaryModel, messages, {
        recentMessageCount,
        compactionPrompt,
      });

      if (onCompaction) {
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

      return { ...params, prompt: compactedPrompt };
    },
  };
}
