/**
 * Built-in lookup table mapping model ID patterns to context window sizes (in tokens).
 * Entries are checked in order; first match wins.
 */
const MODEL_CONTEXT_SIZES: Array<{ pattern: RegExp; tokens: number }> = [
  // OpenAI
  { pattern: /gpt-4\.1/, tokens: 1_047_576 },
  { pattern: /gpt-4o/, tokens: 128_000 },
  { pattern: /gpt-4-turbo/, tokens: 128_000 },
  { pattern: /gpt-4-0125/, tokens: 128_000 },
  { pattern: /gpt-4-1106/, tokens: 128_000 },
  { pattern: /gpt-4$/, tokens: 8_192 },
  { pattern: /gpt-3\.5-turbo-16k/, tokens: 16_384 },
  { pattern: /gpt-3\.5-turbo/, tokens: 16_384 },
  { pattern: /o1-mini/, tokens: 128_000 },
  { pattern: /o1-preview/, tokens: 128_000 },
  { pattern: /o1(?!-)/, tokens: 200_000 },
  { pattern: /o3-mini/, tokens: 200_000 },
  { pattern: /o3(?!-)/, tokens: 200_000 },
  { pattern: /o4-mini/, tokens: 200_000 },

  // Anthropic
  { pattern: /claude-3-5-sonnet/, tokens: 200_000 },
  { pattern: /claude-3-5-haiku/, tokens: 200_000 },
  { pattern: /claude-3-opus/, tokens: 200_000 },
  { pattern: /claude-3-sonnet/, tokens: 200_000 },
  { pattern: /claude-3-haiku/, tokens: 200_000 },
  { pattern: /claude-sonnet-4/, tokens: 200_000 },
  { pattern: /claude-opus-4/, tokens: 200_000 },
  { pattern: /claude-haiku-4/, tokens: 200_000 },
  { pattern: /claude/, tokens: 200_000 },

  // Google
  { pattern: /gemini-2/, tokens: 1_048_576 },
  { pattern: /gemini-1\.5-pro/, tokens: 2_097_152 },
  { pattern: /gemini-1\.5-flash/, tokens: 1_048_576 },
  { pattern: /gemini-1\.0/, tokens: 32_768 },
  { pattern: /gemini-pro/, tokens: 32_768 },

  // Meta Llama
  { pattern: /llama-3\.3/, tokens: 128_000 },
  { pattern: /llama-3\.2/, tokens: 128_000 },
  { pattern: /llama-3\.1/, tokens: 128_000 },
  { pattern: /llama-3/, tokens: 8_192 },
  { pattern: /llama-2/, tokens: 4_096 },

  // Mistral
  { pattern: /mistral-large/, tokens: 128_000 },
  { pattern: /mistral-medium/, tokens: 32_000 },
  { pattern: /mistral-small/, tokens: 128_000 },
  { pattern: /mixtral/, tokens: 32_768 },
  { pattern: /mistral/, tokens: 32_768 },

  // Cohere
  { pattern: /command-r-plus/, tokens: 128_000 },
  { pattern: /command-r/, tokens: 128_000 },

  // DeepSeek
  { pattern: /deepseek/, tokens: 128_000 },
];

/**
 * Look up the context window size for a given model ID.
 * Returns undefined if the model is not recognized.
 */
export function getModelContextSize(modelId: string): number | undefined {
  const id = modelId.toLowerCase();
  for (const { pattern, tokens } of MODEL_CONTEXT_SIZES) {
    if (pattern.test(id)) {
      return tokens;
    }
  }
  return undefined;
}
