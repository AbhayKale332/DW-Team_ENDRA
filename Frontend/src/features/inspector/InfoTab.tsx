import { Alert, Anchor, List, Stack, Table, Text } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { ElevationReference } from '@/features/anchoring/ElevationReference';
import { lonLatAt, formatLonLat } from '@/lib/georef';
import { summariseObjects } from '@/lib/objects';

function Rows({ rows }: { rows: Array<[string, React.ReactNode]> }) {
  return (
    <Table withRowBorders={false} verticalSpacing={3} horizontalSpacing={0} fz="xs">
      <Table.Tbody>
        {rows.map(([k, v]) => (
          <Table.Tr key={k}>
            <Table.Td c="dimmed" w="42%" valign="top">
              {k}
            </Table.Td>
            <Table.Td className="dw-mono" ta="right">
              {v}
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}

const metres = (v: number | null) => (v === null ? '–' : `${v.toFixed(1)} m`);

/** Scene metadata, provenance, model status lines, notes and downloadable model artefacts. */
export function InfoTab() {
  const scene = useScene((s) => s.scene);
  if (!scene) return <Text size="sm" c="dimmed" p="md">No scene loaded.</Text>;
  const s = scene.stats;
  const g = scene.georef;
  const centre = g ? lonLatAt(g, (scene.heights.width - 1) / 2, (scene.heights.height - 1) / 2) : null;
  const notes = scene.warnings.filter((w) => w.level === 'info');
  const obj = scene.objects ? summariseObjects(scene.objects) : null;
  return (
    <Stack gap="md" p="md">
      <ElevationReference scene={scene} />
      <div>
        <span className="dw-section-title">{scene.product === 'DSM' ? 'Elevations' : 'Heights'}</span>
        <Rows
          rows={[
            ['Minimum', `${s.min.toFixed(2)} m`],
            ['Maximum', `${s.max.toFixed(2)} m`],
            ['Mean', `${s.mean.toFixed(2)} m`],
            ['Median', `${s.median.toFixed(2)} m`],
            ['2nd / 98th percentile', `${s.p2.toFixed(2)} / ${s.p98.toFixed(2)} m`],
            ['Below 1 m (above ground)', `${(s.fracBelow1m * 100).toFixed(1)} %`],
          ]}
        />
      </div>
      <div>
        <span className="dw-section-title">Geometry</span>
        <Rows
          rows={[
            ['Height grid', `${scene.heights.width} × ${scene.heights.height} px`],
            ['Ground resolution', `${scene.gsd.toFixed(3)} m/px (${scene.gsdSource})`],
            ['Footprint', `${((scene.heights.width * scene.gsd) / 1000).toFixed(3)} × ${((scene.heights.height * scene.gsd) / 1000).toFixed(3)} km`],
            ['Input image', `${scene.imageWidth} × ${scene.imageHeight} px`],
            ['CRS', g ? (g.epsg ? `EPSG:${g.epsg}` : 'unknown') : 'not georeferenced'],
            ...(centre ? ([['Scene centre', formatLonLat(centre)]] as Array<[string, string]>) : []),
          ]}
        />
      </div>
      {obj && (
        <div>
          <span className="dw-section-title">Detected objects</span>
          <Rows
            rows={[
              ['Trees', obj.trees ? `${obj.trees} · mean ${metres(obj.treeMeanH)} · tallest ${metres(obj.treeMaxH)}` : '0'],
              ['Buildings', obj.buildings ? `${obj.buildings} · median ${metres(obj.buildingMedianH)} · tallest ${metres(obj.buildingMaxH)}` : '0'],
              ['Water bodies', String(obj.water)],
            ]}
          />
          {obj.truncated && (
            <Text size="xs" c="dimmed" mt={4}>
              The model capped the object count for this scene; the smallest objects were dropped.
            </Text>
          )}
        </div>
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
      <div>
        <span className="dw-section-title">Provenance</span>
        <Rows
          rows={[
            ['Source', scene.provenance.provider],
            ['Created', new Date(scene.provenance.createdAt).toLocaleString()],
            ...(scene.provenance.params ? ([['TTA', scene.provenance.params.tta ? 'on (8 passes)' : 'off']] as Array<[string, string]>) : []),
            ...(scene.provenance.modelVersion ? ([['Model', scene.provenance.modelVersion]] as Array<[string, string]>) : []),
          ]}
        />
      </div>
      {scene.statusLines.length > 0 && (
        <div>
          <span className="dw-section-title">Model report</span>
          <Stack gap={2} mt={4}>
            {scene.statusLines.map((l) => (
              <Text key={l} size="xs" className="dw-mono">
                {l.replace(/\*\*|`/g, '')}
              </Text>
            ))}
          </Stack>
        </div>
      )}
      {scene.artefacts.length > 0 && (
        <div>
          <span className="dw-section-title">Model output files</span>
          <List size="xs" mt={4} spacing={2}>
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
          <Text size="xs" c="dimmed" mt={4}>
            .npy = raw metres · .glb = textured mesh · meta.json = encoding and scene metadata · seg.png = class per pixel · objects.json = trees, buildings and water · .zip = everything incl. OBJ.
          </Text>
        </div>
      )}
    </Stack>
  );
}
