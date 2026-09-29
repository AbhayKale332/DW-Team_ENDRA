import { describe, expect, it } from 'vitest';
import { createRateLimiter, isAllowedSpaceRequest, pickForwardHeaders } from './guard.mjs';

describe('isAllowedSpaceRequest', () => {
  it('lets the app’s Gradio calls through', () => {
    expect(isAllowedSpaceRequest('GET', '/config')).toBe(true);
    expect(isAllowedSpaceRequest('POST', '/gradio_api/upload')).toBe(true);
    expect(isAllowedSpaceRequest('POST', '/gradio_api/call/predict')).toBe(true);
    expect(isAllowedSpaceRequest('GET', '/gradio_api/call/predict/abc123_x')).toBe(true);
    expect(isAllowedSpaceRequest('GET', '/gradio_api/file=/tmp/gradio/ab12/ndsm_m.npy')).toBe(true);
    expect(isAllowedSpaceRequest('HEAD', '/config')).toBe(true);
  });

  it('refuses everything else', () => {
    expect(isAllowedSpaceRequest('POST', '/config')).toBe(false);
    expect(isAllowedSpaceRequest('GET', '/')).toBe(false);
    expect(isAllowedSpaceRequest('POST', '/gradio_api/call/other_fn')).toBe(false);
    expect(isAllowedSpaceRequest('POST', '/gradio_api/queue/join')).toBe(false);
    expect(isAllowedSpaceRequest('DELETE', '/gradio_api/upload')).toBe(false);
    expect(isAllowedSpaceRequest('GET', '/gradio_api/file=../../etc/passwd')).toBe(false);
  });
});

describe('pickForwardHeaders', () => {
  it('drops cookies, origin and client identity', () => {
    const out = pickForwardHeaders({ cookie: 'a=b', origin: 'https://x', 'x-forwarded-for': '1.2.3.4', authorization: 'Bearer user', 'content-type': 'application/json', accept: '*/*' });
    expect(out).toEqual({ 'content-type': 'application/json', accept: '*/*' });
  });
});

describe('createRateLimiter', () => {
  it('allows max hits per window, then reports the wait', () => {
    let t = 0;
    const rl = createRateLimiter({ max: 2, windowMs: 1000, now: () => t });
    expect(rl.hit('a')).toBe(0);
    expect(rl.hit('a')).toBe(0);
    expect(rl.hit('a')).toBe(1000);
    expect(rl.hit('b')).toBe(0);
    t = 1000;
    expect(rl.hit('a')).toBe(0);
  });
});
