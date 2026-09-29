export declare const OVERPASS_MIRRORS: Record<string, string>;
export declare function relayOverpass(
  mirror: string,
  body: string,
  site: string | undefined,
): Promise<{ status: number; contentType: string; body: string }>;
