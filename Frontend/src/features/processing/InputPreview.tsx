import { isPreviewing, useScene } from '@/store/scene';
import classes from '@/features/viewport/overlays/overlays.module.css';

/** The input image over the viewport while a run is in progress; the 3D result replaces it as soon as it exists. */
export function InputPreview() {
  const previewing = useScene(isPreviewing);
  const url = useScene((s) => s.input?.previewUrl);
  const name = useScene((s) => s.input?.name);
  if (!previewing || !url) return null;
  return (
    <div className={classes.preview}>
      <img src={url} alt={name ? `Input image ${name}` : 'Input image'} className={classes.previewImage} />
    </div>
  );
}
