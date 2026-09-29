/** "owner/Space_Name" → "https://owner-space-name.hf.space" (Hugging Face direct-URL convention). */
export function spaceUrlFromId(id) {
  const sub = String(id).trim().replace('/', '-').replace(/[._]/g, '-').toLowerCase();
  return `https://${sub}.hf.space`;
}
