import { EventEmitter } from 'node:events';
import { describe, expect, it } from 'vitest';
import { createTokenPool, parseTokens, quotaRetryMs } from './tokens.mjs';

function stream(...chunks) {
  const res = new EventEmitter();
  return { res, play: () => (chunks.forEach((c) => res.emit('data', Buffer.from(c))), res.emit('end')) };
}

describe('parseTokens', () => {
  it('reads HF_TOKENS, HF_TOKEN and numbered tokens in order, without duplicates', () => {
    expect(parseTokens({ HF_TOKENS: 'a, b', HF_TOKEN: 'c', HF_TOKEN_10: 'e', HF_TOKEN_2: 'd', HF_TOKEN_3: 'a' })).toEqual(['a', 'b', 'c', 'd', 'e']);
    expect(parseTokens({})).toEqual([]);
  });
});

describe('quotaRetryMs', () => {
  it('parses the ZeroGPU reset time', () => {
    expect(quotaRetryMs('You have exceeded your GPU quota (120s requested vs. 44s left). Try again in 1:02:03')).toBe(3723_000);
    expect(quotaRetryMs('Your ZeroGPU quota for today is used up.')).toBe(0);
    expect(quotaRetryMs('event: complete')).toBeNull();
  });
});

describe('createTokenPool', () => {
  it('benches a token whose run hit the quota and keeps the run on the token it started with', () => {
    let t = 0;
    const pool = createTokenPool(['a', 'b'], { cooldownMs: 1000, now: () => t, log: () => {} });
    expect(pool.pick('/gradio_api/call/predict')).toBe('a');
    expect(pool.spare('a')).toBe(1);

    const call = stream('{"event_', 'id":"ev1"}');
    pool.observe('/gradio_api/call/predict', 'a', call.res);
    call.play();
    expect(pool.pick('/gradio_api/call/predict/ev1')).toBe('a');

    const result = stream('event: error\ndata: "You have exceeded your GPU qu', 'ota. Try again in 0:00:05"\n\n');
    pool.observe('/gradio_api/call/predict/ev1', 'a', result.res);
    result.play();
    expect(pool.pick('/gradio_api/call/predict')).toBe('b');
    expect(pool.spare('b')).toBe(0);

    t = 5000;
    expect(pool.pick('/gradio_api/call/predict')).toBe('a');
  });

  it('falls back to the token that recovers first when all are benched', () => {
    const pool = createTokenPool(['a', 'b'], { now: () => 0, log: () => {} });
    for (const [token, wait] of [['a', '0:00:09'], ['b', '0:00:03']]) {
      const s = stream(`data: "GPU quota exceeded. Try again in ${wait}"`);
      pool.observe('/gradio_api/call/predict/x', token, s.res);
      s.play();
    }
    expect(pool.pick('/config')).toBe('b');
  });

  it('benches a token the Space refused and moves on to the next one', () => {
    const pool = createTokenPool(['a', 'b', 'c'], { now: () => 0, log: () => {} });
    for (const [token, statusCode, path] of [['a', 401, '/gradio_api/upload'], ['b', 404, '/config']]) {
      const res = new EventEmitter();
      Object.assign(res, { statusCode, headers: {} });
      pool.observe(path, token, res);
    }
    expect(pool.pick('/config')).toBe('c');
    expect(pool.spare('c')).toBe(0);
  });

  it('benches a token whose run reported any error, not only quota', () => {
    let t = 0;
    const pool = createTokenPool(['a', 'b'], { failCooldownMs: 1000, now: () => t, log: () => {} });
    const s = stream('event: heartbeat\ndata: null\n\nevent: error\ndata: "ZeroGPU worker error"\n\n');
    pool.observe('/gradio_api/call/predict/ev2', 'a', s.res);
    s.play();
    expect(pool.pick('/gradio_api/call/predict')).toBe('b');
    t = 1000;
    expect(pool.pick('/gradio_api/call/predict')).toBe('a');
  });
});
