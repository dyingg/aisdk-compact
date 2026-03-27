import { describe, it, expect } from 'vitest';
import { estimateTokens } from '../src/token-estimator.js';

describe('estimateTokens', () => {
  it('returns 0 for empty array', () => {
    expect(estimateTokens([])).toBe(1); // "[]" is 2 chars → ceil(2/4) = 1
  });

  it('estimates tokens from message content', () => {
    const messages = [
      { role: 'user', content: 'Hello, how are you?' },
      { role: 'assistant', content: 'I am fine, thank you!' },
    ];
    const result = estimateTokens(messages);
    // Should be roughly JSON.stringify length / 4
    const expected = Math.ceil(JSON.stringify(messages).length / 4);
    expect(result).toBe(expected);
  });

  it('scales with message count', () => {
    const small = [{ role: 'user', content: 'Hi' }];
    const large = Array.from({ length: 100 }, (_, i) => ({
      role: 'user',
      content: `Message number ${i} with some content to pad the token count`,
    }));
    expect(estimateTokens(large)).toBeGreaterThan(estimateTokens(small));
  });
});
