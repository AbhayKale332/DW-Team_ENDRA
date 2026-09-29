/** Display formatting shared by the panels, so the same quantity always reads the same way. */

/** 0.123 → "12 %"; shares under 10 % keep one decimal so small values do not collapse to 0. */
export const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)} %` : '—');

/** Square metres → m², ha or km². */
export const area = (m2: number) => (m2 >= 1e6 ? `${(m2 / 1e6).toFixed(2)} km²` : m2 >= 1e4 ? `${(m2 / 1e4).toFixed(1)} ha` : `${m2.toFixed(0)} m²`);

/** Metres with a fixed number of decimals. */
export const metres = (v: number, digits = 1) => (Number.isFinite(v) ? `${v.toFixed(digits)} m` : '—');
