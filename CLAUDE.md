# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
bun install          # Install dependencies
bun run build        # Build ESM + CJS with type declarations (tsup)
bun run typecheck    # TypeScript type checking (tsc --noEmit)
bun test             # Run all tests (vitest via bun)
bun test <file>      # Run a single test file, e.g. bun test test/compact.test.ts
```

## Architecture

This is a Vercel AI SDK v6 middleware library. It exports `compactMiddleware()` which returns a `LanguageModelV3Middleware` that users wrap around any model via `wrapLanguageModel`.

**Flow:** On every generate/stream call, the middleware's `transformParams` hook estimates the token count of `params.prompt`. If it exceeds `threshold * maxTokens`, it splits messages into system/older/recent, calls the model (or a separate `compactionModel`) with a summarization prompt, and replaces older messages with a single summary message.

### Source modules

- `src/middleware.ts` — `compactMiddleware()` factory. The only middleware hook used is `transformParams` (the only hook that can modify messages before they reach the model).
- `src/compact.ts` — `compactMessages()` core logic: splits prompt into system/older/recent, serializes older messages, calls `model.doGenerate()` for summarization, returns compacted prompt.
- `src/model-context.ts` — Built-in model ID → context window size lookup table. Pattern-matched against `model.modelId`. Used when `maxTokens` is not explicitly provided.
- `src/token-estimator.ts` — Default token estimation (`JSON.stringify(messages).length / 4`).
- `src/prompt.ts` — Default compaction system prompt.
- `src/types.ts` — `CompactMiddlewareOptions` and `CompactionInfo` types.

### Key types from AI SDK v6

The middleware targets `@ai-sdk/provider` types: `LanguageModelV3`, `LanguageModelV3Middleware`, `LanguageModelV3CallOptions`, `LanguageModelV3Prompt` (array of `LanguageModelV3Message`). The `LanguageModelMiddleware` alias in the `ai` package maps to `LanguageModelV3Middleware`. Tool call parts use `input`/`output` (not `args`/`result` from V1).

### Peer dependency

`ai@^6` is a peer dependency. `@ai-sdk/provider` types come transitively. Tests use `MockLanguageModelV3` from `ai/test`.
