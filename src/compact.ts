import type {
  LanguageModelV3,
  LanguageModelV3Message,
  LanguageModelV3Prompt,
} from "@ai-sdk/provider";
import { defaultCompactionPrompt } from "./prompt.js";

/**
 * Serialize messages into a text representation for the compaction prompt.
 */
function serializeMessages(messages: LanguageModelV3Message[]): string {
  return messages
    .map((msg) => {
      switch (msg.role) {
        case "system":
          return `[system]: ${msg.content}`;
        case "user":
        case "assistant": {
          const text = msg.content
            .map((part) => {
              switch (part.type) {
                case "text":
                  return part.text;
                case "reasoning":
                  return `[reasoning]: ${part.text}`;
                case "tool-call":
                  return `[tool-call: ${part.toolName}(${JSON.stringify(part.input)})]`;
                case "tool-result":
                  return `[tool-result: ${part.toolName} -> ${JSON.stringify(part.output)}]`;
                default:
                  return `[${part.type}]`;
              }
            })
            .join("\n");
          return `[${msg.role}]: ${text}`;
        }
        case "tool": {
          const text = msg.content
            .map((part) => {
              switch (part.type) {
                case "tool-result":
                  return `[tool-result: ${part.toolName} -> ${JSON.stringify(part.output)}]`;
                default:
                  return `[${part.type}]`;
              }
            })
            .join("\n");
          return `[tool]: ${text}`;
        }
      }
    })
    .join("\n\n");
}

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

  // Serialize older messages for the summarization prompt
  const serialized = serializeMessages(olderMessages);

  const summaryPrompt = compactionPrompt ?? defaultCompactionPrompt;

  if (debug) {
    console.debug(
      `[aisdk-compact] Summarizing ${olderMessages.length} older messages (keeping ${recentMessages.length} recent)`
    );
  }

  // Call the model to generate a summary
  const result = await model.doGenerate({
    prompt: [
      { role: "system", content: summaryPrompt },
      {
        role: "user",
        content: [
          {
            type: "text",
            text: `Here is the conversation history to compact:\n\n${serialized}`,
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

  console.log([...systemMessages, summaryMessage, ...recentMessages]);

  return [...systemMessages, summaryMessage, ...recentMessages];
}
