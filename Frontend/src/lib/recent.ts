import { createStore, del, get, set } from 'idb-keyval';

export interface RecentEntry {
  id: string;
  name: string;
  savedAt: string;
  thumbnail: string | null;
  size: number;
}

const store = createStore('depthwizard', 'recent');
const INDEX = 'index';
const MAX = 10;

export async function listRecent(): Promise<RecentEntry[]> {
  try {
    return ((await get<RecentEntry[]>(INDEX, store)) ?? []).slice(0, MAX);
  } catch {
    return [];
  }
}

export async function putRecent(entry: RecentEntry, blob: Blob) {
  try {
    const list = (await listRecent()).filter((e) => e.id !== entry.id);
    list.unshift(entry);
    const drop = list.splice(MAX);
    await set(`blob:${entry.id}`, blob, store);
    await set(INDEX, list, store);
    await Promise.all(drop.map((d) => del(`blob:${d.id}`, store)));
  } catch (e) {
    console.warn('Recent projects unavailable', e);
  }
}

export async function getRecentBlob(id: string): Promise<Blob | undefined> {
  return get<Blob>(`blob:${id}`, store);
}

export async function removeRecent(id: string) {
  const list = (await listRecent()).filter((e) => e.id !== id);
  await set(INDEX, list, store);
  await del(`blob:${id}`, store);
}
