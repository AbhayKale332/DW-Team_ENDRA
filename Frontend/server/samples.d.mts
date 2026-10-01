export interface SampleEntry {
  id: string;
  name: string;
  /** Path of the .dwproj relative to the samples directory, forward slashes. */
  file: string;
  /** Input image and height map shown while the project downloads (scripts/sample-previews.mjs), same base. */
  image?: string;
  height?: string;
}
export declare function listSamples(dir: string): SampleEntry[];
export declare function samplesIndex(dir: string): string;
