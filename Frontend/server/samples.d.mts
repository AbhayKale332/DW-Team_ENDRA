export interface SampleEntry {
  id: string;
  name: string;
  /** Path of the .dwproj relative to the samples directory, forward slashes. */
  file: string;
}
export declare function listSamples(dir: string): SampleEntry[];
export declare function samplesIndex(dir: string): string;
