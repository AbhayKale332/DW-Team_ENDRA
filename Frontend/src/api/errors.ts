/** Typed failures. The Gradio Space reports most failures as a *successful* response whose status
 *  text describes the problem, so these are produced by classifying that text. */

export type ErrorKind = 'quota' | 'no-image' | 'inference' | 'network' | 'unavailable' | 'auth' | 'cancelled' | 'invalid-input';

export class DepthWizardError extends Error {
  constructor(
    public readonly kind: ErrorKind,
    message: string,
    public readonly detail?: string,
  ) {
    super(message);
    this.name = 'DepthWizardError';
  }
}

export const ERROR_COPY: Record<ErrorKind, { title: string; hint: string }> = {
  quota: {
    title: 'GPU quota used up',
    hint: 'The ZeroGPU allowance of every configured HF token is exhausted for today. Retry later, turn TTA off, or open a sample scene.',
  },
  'no-image': { title: 'No image received', hint: 'Open an image and run again.' },
  inference: { title: 'Inference failed', hint: 'The model could not process this image. Check the file and the declared GSD, then retry.' },
  network: { title: 'Connection problem', hint: 'The model service could not be reached. Check your connection and retry.' },
  unavailable: {
    title: 'Model service unavailable',
    hint: 'The Space is sleeping, building or paused. It can take about a minute to wake up — retry shortly.',
  },
  auth: {
    title: 'Access denied',
    hint: 'The private model Space rejected the request. Set a valid HF_TOKEN (read access to the Space) in .env and restart the server.',
  },
  cancelled: { title: 'Run cancelled', hint: 'The request was cancelled.' },
  'invalid-input': { title: 'Unsupported input', hint: 'Use a PNG, JPG or (Geo)TIFF RGB image.' },
};

/** Classify the Space's status markdown when no artefacts were returned. */
export function classifyStatusText(status: string): DepthWizardError {
  const s = status.toLowerCase();
  if (s.includes('quota') || s.includes('gpu task aborted') || s.includes('gpu time')) return new DepthWizardError('quota', ERROR_COPY.quota.title, status);
  if (s.includes('upload an image first')) return new DepthWizardError('no-image', ERROR_COPY['no-image'].title, status);
  if (s.includes('inference failed')) {
    const detail = status.replace(/^.*?inference failed:\s*/i, '');
    return new DepthWizardError('inference', ERROR_COPY.inference.title, detail || status);
  }
  return new DepthWizardError('inference', ERROR_COPY.inference.title, status || 'The model returned no result.');
}

/** Map thrown errors (network, client, auth) to typed errors. */
export function toDepthWizardError(e: unknown): DepthWizardError {
  if (e instanceof DepthWizardError) return e;
  const msg = e instanceof Error ? e.message : typeof e === 'object' && e && 'message' in e ? String((e as { message: unknown }).message) : String(e);
  const m = msg.toLowerCase();
  if (m.includes('abort') || m.includes('cancel')) return new DepthWizardError('cancelled', ERROR_COPY.cancelled.title, msg);
  if (m.includes('401') || m.includes('403') || m.includes('unauthor') || m.includes('credentials') || m.includes('invalid username'))
    return new DepthWizardError('auth', ERROR_COPY.auth.title, msg);
  if (m.includes('quota')) return new DepthWizardError('quota', ERROR_COPY.quota.title, msg);
  if (m.includes('sleep') || m.includes('building') || m.includes('paused') || m.includes('404') || m.includes('not found') || m.includes('could not resolve app config'))
    return new DepthWizardError('unavailable', ERROR_COPY.unavailable.title, msg);
  if (m.includes('fetch') || m.includes('network') || m.includes('failed to')) return new DepthWizardError('network', ERROR_COPY.network.title, msg);
  return new DepthWizardError('inference', ERROR_COPY.inference.title, msg);
}
