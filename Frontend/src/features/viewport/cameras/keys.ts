import { useEffect, useRef } from 'react';

const isTyping = (e: KeyboardEvent) => {
  const t = e.target as HTMLElement | null;
  return !!t && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
};

/** Live set of pressed keys (by `KeyboardEvent.code`), ignoring keys typed into form fields. */
export function usePressedKeys(capture: string[] = []) {
  const keys = useRef(new Set<string>());
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (isTyping(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      keys.current.add(e.code);
      if (capture.includes(e.code)) e.preventDefault();
    };
    const up = (e: KeyboardEvent) => keys.current.delete(e.code);
    const blur = () => keys.current.clear();
    window.addEventListener('keydown', down);
    window.addEventListener('keyup', up);
    window.addEventListener('blur', blur);
    return () => {
      window.removeEventListener('keydown', down);
      window.removeEventListener('keyup', up);
      window.removeEventListener('blur', blur);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return keys;
}

export function headingOf(dirX: number, dirZ: number) {
  // Forward vector (x east, z south) → compass bearing clockwise from north.
  const deg = (Math.atan2(dirX, -dirZ) * 180) / Math.PI;
  return (deg + 360) % 360;
}
