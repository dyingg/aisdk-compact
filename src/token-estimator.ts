/**
 * Default token estimation: approximate tokens from JSON-serialized message length.
 * Roughly 1 token per 4 characters for English text.
 */
export function estimateTokens(messages: unknown[]): number {
  return Math.ceil(JSON.stringify(messages).length / 4);
}
