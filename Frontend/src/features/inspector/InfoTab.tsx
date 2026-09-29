import { Alert, Anchor, List, Stack, Text } from '@mantine/core';
import { IconInfoCircle, IconPhoto } from '@tabler/icons-react';
import { EmptyPanel, KeyValueRows, PanelSection } from '@/components/panel';
import { useScene } from '@/store/scene';
import { ElevationReference } from '@/features/anchoring/ElevationReference';
import { lonLatAt, formatLonLat } from '@/lib/georef';
import { summariseObjects } from '@/lib/objects';

const metres = (v: number | null) => (v === null ? '–' : `${v.toFixed(1)} m`);

/** Scene metadata, provenance, model status lines, notes and downloadable model artefacts. */
export function InfoTab() {
  const scene = useScene((s) => s.scene);
  if (!scene) return <EmptyPanel icon={IconPhoto}>No scene loaded.</EmptyPanel>;
  const s = scene.stats;
  const g = scene.georef;
  const centre = g ? lonLatAt(g, (scene.heights.width - 1) / 2, (scene.heights.height - 1) / 2) : null;
  const notes = scene.warnings.filter((w) => w.level === 'info');
  const obj = scene.objects ? summariseObjects(scene.objects) : null;
  return (
    <Stack gap="lg" p="md">
      <ElevationReference scene={scene} />
      <PanelSection title={scene.product === 'DSM' ? 'Elevations' : 'Heights'} gap={4}>
        <KeyValueRows
          rows={[
            ['Minimum', `${s.min.toFixed(2)} m`],
            ['Maximum', `${s.max.toFixed(2)} m`],
            ['Mean', `${s.mean.toFixed(2)} m`],
            ['Median', `${s.median.toFixed(2)} m`],
            ['2nd / 98th percentile', `${s.p2.toFixed(2)} / ${s.p98.toFixed(2)} m`],
            ['Below 1 m (above ground)', `${(s.fracBelow1m * 100).toFixed(1)} %`],
          ]}
        />
      </PanelSection>
      <PanelSection title="Geometry" gap={4}>
        <KeyValueRows
          rows={[
            ['Height grid', `${scene.heights.width} × ${scene.heights.height} px`],
            ['Ground resolution', `${scene.gsd.toFixed(3)} m/px (${scene.gsdSource})`],
            ['Footprint', `${((scene.heights.width * scene.gsd) / 1000).toFixed(3)} × ${((scene.heights.height * scene.gsd) / 1000).toFixed(3)} km`],
            ['Input image', `${scene.imageWidth} × ${scene.imageHeight} px`],
            ['CRS', g ? (g.epsg ? `EPSG:${g.epsg}` : 'unknown') : 'not georeferenced'],
            ...(centre ? ([['Scene centre', formatLonLat(centre)]] as Array<[string, string]>) : []),
          ]}
        />
      </PanelSection>
      {obj && (
        <PanelSection title="Detected objects" gap={4}>
          <KeyValueRows
            rows={[
              ['Trees', obj.trees ? `${obj.trees} · mean ${metres(obj.treeMeanH)} · tallest ${metres(obj.treeMaxH)}` : '0'],
              ['Buildings', obj.buildings ? `${obj.buildings} · median ${metres(obj.buildingMedianH)} · tallest ${metres(obj.buildingMaxH)}` : '0'],
              ['Water bodies', String(obj.water)],
            ]}
          />
          {obj.truncated && (
            <Text size="xs" c="dimmed">
              Capped by the model: the smallest objects were dropped.
            </Text>
          )}
        </PanelSection>
      )}
      {notes.length > 0 && (
        <Alert variant="light" color="gray" icon={<IconInfoCircle size={16} />} p="xs">
          <List size="xs" spacing={4}>
            {notes.map((n) => (
              <List.Item key={n.id}>
                <b>{n.title}.</b> {n.message}
              </List.Item>
            ))}
          </List>
        </Alert>
      )}
      <PanelSection title="Provenance" gap={4}>
        <KeyValueRows
          rows={[
            ['Source', scene.provenance.provider],
            ['Created', new Date(scene.provenance.createdAt).toLocaleString()],
            ...(scene.provenance.params ? ([['TTA', scene.provenance.params.tta ? 'on (8 passes)' : 'off']] as Array<[string, string]>) : []),
            ...(scene.provenance.modelVersion ? ([['Model', scene.provenance.modelVersion]] as Array<[string, string]>) : []),
          ]}
        />
      </PanelSection>
      {scene.statusLines.length > 0 && (
        <PanelSection title="Model report" gap={4}>
          <Stack gap={2}>
            {scene.statusLines.map((l) => (
              <Text key={l} size="xs" className="dw-mono">
                {l.replace(/\*\*|`/g, '')}
              </Text>
            ))}
          </Stack>
        </PanelSection>
      )}
      {scene.artefacts.length > 0 && (
        <PanelSection title="Model output files" gap={4}>
          <List size="xs" spacing={2}>
            {scene.artefacts.map((a) => (
              <List.Item key={a.name}>
                {a.url ? (
                  <Anchor href={a.url} target="_blank" rel="noopener" download={a.name}>
                    {a.name}
                  </Anchor>
                ) : (
                  a.name
                )}
                {a.size ? <Text span c="dimmed"> · {(a.size / 1e6).toFixed(1)} MB</Text> : null}
              </List.Item>
            ))}
          </List>
        </PanelSection>
      )}
    </Stack>
  );
}
