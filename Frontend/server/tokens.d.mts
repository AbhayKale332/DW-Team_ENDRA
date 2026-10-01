import type { IncomingMessage } from 'node:http';

export declare function parseTokens(env: Record<string, string | undefined>): string[];
export declare function isWatchedPath(path: string): boolean;
export declare function quotaRetryMs(text: string): number | null;
export declare function isTokenFailure(status: number | undefined, path: string): boolean;
export interface TokenPool {
  size: number;
  pick(path: string): string | undefined;
  spare(token: string | undefined): number;
  observe(path: string, token: string | undefined, res: IncomingMessage): void;
}
export declare function createTokenPool(
  tokens: string[],
  opts?: { cooldownMs?: number; failCooldownMs?: number; now?: () => number; log?: (msg: string) => void },
): TokenPool;
