import type { LanguageModelV3 } from "@ai-sdk/provider";

export interface CompactMiddlewareOptions {
  /**
   * Maximum context window size in tokens.
   * If omitted, auto-resolved from the model ID using a built-in lookup table.
   */
  maxTokens?: number;

  /**
   * Fraction of maxTokens that triggers compaction.
   * @default 0.8
   */
  threshold?: number;

  /**
   * Number of recent messages to always preserve (never compacted).
   * System messages are always preserved separately.
   * @default 6
   */
  recentMessageCount?: number;

  /**
   * Separate model to use for generating summaries.
   * Defaults to the wrapped model itself.
   */
  compactionModel?: LanguageModelV3;

  /**
   * Custom system prompt for the summarization call.
   * If provided, replaces the default compaction prompt entirely.
   */
  compactionPrompt?: string;

  /**
   * Custom token estimation function.
   * Receives the raw message array and should return an estimated token count.
   * @default JSON.stringify char length / 4
   */
  estimateTokens?: (messages: unknown[]) => number;

  /**
   * Callback fired after compaction occurs.
   */
  onCompaction?: (info: CompactionInfo) => void;

  /**
   * Enable verbose debug logging of the compaction flow.
   * Logs are prefixed with [aisdk-compact] and sent to console.debug.
   * @default false
   */
  debug?: boolean;
}

export interface CompactionInfo {
  originalTokens: number;
  compactedTokens: number;
  removedMessageCount: number;
}
