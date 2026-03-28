import type {
  LanguageModelV3,
  LanguageModelV3Message,
  LanguageModelV3Prompt,
} from "@ai-sdk/provider";
import { defaultCompactionPrompt } from "./prompt.js";

export interface CompactMessagesOptions {
  recentMessageCount: number;
  compactionPrompt?: string;
  debug?: boolean;
}

/**
 * Compact a message array by summarizing older messages via the given model.
 *
 * Returns the compacted prompt: [system messages] + [summary message] + [recent messages].
 */
export async function compactMessages(
  model: LanguageModelV3,
  prompt: LanguageModelV3Prompt,
  options: CompactMessagesOptions
): Promise<LanguageModelV3Prompt> {
  const { recentMessageCount, compactionPrompt, debug = false } = options;

  // Separate system messages from the rest
  const systemMessages: LanguageModelV3Message[] = [];
  const conversationMessages: LanguageModelV3Message[] = [];

  for (const msg of prompt) {
    if (msg.role === "system") {
      systemMessages.push(msg);
    } else {
      conversationMessages.push(msg);
    }
  }

  // If not enough messages to split, return as-is
  if (conversationMessages.length <= recentMessageCount) {
    if (debug) {
      console.debug(
        `[aisdk-compact] Skipping: only ${conversationMessages.length} conversation messages, need >${recentMessageCount} to split`
      );
    }
    return prompt;
  }

  const excessMessagesCount = conversationMessages.length - recentMessageCount;
  const olderMessages = conversationMessages.slice(0, excessMessagesCount);
  const recentMessages = conversationMessages.slice(excessMessagesCount);

  const summaryPrompt = compactionPrompt ?? defaultCompactionPrompt;

  if (debug) {
    console.debug(
      `[aisdk-compact] Summarizing ${olderMessages.length} older messages (keeping ${recentMessages.length} recent)`
    );
  }

  // Call the model with real message objects + compaction instruction as a final user message
  const result = await model.doGenerate({
    prompt: [
      ...systemMessages,
      ...olderMessages,
      {
        role: "user",
        content: [
          {
            type: "text",
            text: summaryPrompt,
          },
        ],
      },
    ],
    providerOptions: {
      "aisdk-compact": {
        purpose: "summarization",
        olderMessageCount: olderMessages.length,
        recentMessageCount: recentMessages.length,
      },
    },
    headers: {
      "X-Aisdk-Compact-Purpose": "summarization",
    },
  });

  // Extract summary text from the response
  const summaryText = result.content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text)
    .join("");

  if (debug) {
    console.debug(`[aisdk-compact] Summary generated (${summaryText.length} chars)`);
    console.debug(
      `[aisdk-compact] Summary: ${summaryText.slice(0, 500)}${summaryText.length > 500 ? "..." : ""}`
    );
  }
  const originalMessages = conversationMessages.length;
  // One synthetic assistant summary + preserved recent conversation messages
  const messagesAfterCompaction = 1 + recentMessages.length;

  // Build the compacted prompt
  const summaryMessage: LanguageModelV3Message = {
    role: "assistant",
    content: [
      {
        type: "text",
        text:
          `[Compacted conversation summary]\n\n` +
          `originalMessages: ${originalMessages}\n` +
          `messagesAfterCompaction: ${messagesAfterCompaction}\n\n` +
          summaryText,
      },
    ],
  };

  return [...systemMessages, summaryMessage, ...recentMessages];
}
