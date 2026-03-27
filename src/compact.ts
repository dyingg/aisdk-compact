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
  const { recentMessageCount, compactionPrompt } = options;

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
    return prompt;
  }

  const olderMessages = conversationMessages.slice(
    0,
    conversationMessages.length - recentMessageCount
  );
  const recentMessages = conversationMessages.slice(
    conversationMessages.length - recentMessageCount
  );

  // Serialize older messages for the summarization prompt
  const serialized = serializeMessages(olderMessages);

  const summaryPrompt = compactionPrompt ?? defaultCompactionPrompt;

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
  });

  // Extract summary text from the response
  const summaryText = result.content
    .filter((c): c is { type: "text"; text: string } => c.type === "text")
    .map((c) => c.text)
    .join("");

  // Build the compacted prompt
  const summaryMessage: LanguageModelV3Message = {
    role: "assistant",
    content: [
      {
        type: "text",
        text: `[Compacted conversation summary]\n\n${summaryText}`,
      },
    ],
  };

  return [...systemMessages, summaryMessage, ...recentMessages];
}
