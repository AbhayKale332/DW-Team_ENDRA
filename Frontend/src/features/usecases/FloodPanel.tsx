import { useEffect, useMemo } from 'react';
import { Alert, Badge, Button, Group, Slider, SegmentedControl, Stack, Switch, Table, Text, Tooltip } from '@mantine/core';
import { IconCrosshair, IconLock, IconPlayerPause, IconPlayerPlay } from '@tabler/icons-react';
import { anchorBlocker } from '@/lib/dem';
import { buildingDepth, floodStats, STOREY_M, TIER_COLORS, TIER_LABELS, TIER_LIMITS, WEIGHTS, type Tier } from '@/lib/usecases/flood';
import { runAnchoring } from '@/features/anchoring/runAnchoring';
import { useAnchor } from '@/store/anchor';
import { useScene } from '@/store/scene';
import { useUseCases, type FloodSourceKind } from '@/store/usecases';
import { runFlood } from './actions';

const area = (m2: number) => (m2 >= 1e6 ? `${(m2 / 1e6).toFixed(2)} km²` : m2 >= 1e4 ? `${(m2 / 1e4).toFixed(2)} ha` : `${m2.toFixed(0)} m²`);

/** Use case 2: flood vulnerability and emergency prioritisation, from DEM-anchored terrain and the model's buildings. */
export function FloodPanel() {
  const scene = useScene((s) => s.scene);
  const anchor = useAnchor();
  const u = useUseCases();
  const { flood, risks, rise } = u;

  // the arrival-level model is built once per scene / source; the slider only compares against it
  useEffect(() => {
    if (scene?.terrain && u.floodStatus === 'idle' && !flood && (u.floodSource !== 'point' || u.floodPoint)) void runFlood();
  }, [scene, u.floodStatus, flood, u.floodSource, u.floodPoint]);

  // rising water animation
  useEffect(() => {
    if (!u.playing || !flood) return;
    let raf = 0;
    let last = performance.now();
    const step = (now: number) => {
      const st = useUseCases.getState();
      const next = st.rise + ((now - last) / 1000) * (flood.maxRise / 14);
      last = now;
      if (next >= flood.maxRise) return void st.set({ rise: flood.maxRise, playing: false });
      st.set({ rise: next });
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [u.playing, flood]);

  const stats = useMemo(() => (flood ? floodStats(flood, rise) : null), [flood, rise]);
  const ranked = useMemo(() => risks.filter((r) => r.tier !== 'none').sort((a, b) => b.score - a.score), [risks]);
  const tierCount = useMemo(() => {
    const c: Record<Tier, number> = { critical: 0, high: 0, watch: 0, none: 0 };
    risks.forEach((r) => c[r.tier]++);
    return c;
  }, [risks]);
  const wetNow = flood ? ranked.filter((r) => r.arrival <= flood.baseLevel + rise).length : 0;

  if (!scene) return null;

  if (!scene.terrain) {
    const blocker = anchorBlocker(scene);
    const running = anchor.sceneId === scene.id && anchor.status === 'running';
    return (
      <Alert color="orange" variant="light" icon={<IconLock size={16} />} title="Needs terrain elevation" p="sm">
        <Stack gap={8}>
          <Text size="xs" lh={1.5}>
            {scene.product === 'rDSM'
              ? 'This is a relative surface model (rDSM). It has no coordinate system and no terrain, so there is no elevation to flood. Nothing is invented: open a georeferenced GeoTIFF to use this analysis.'
              : 'The model predicts height above ground, which is flat by construction. Flood depth needs real terrain, which comes from the DEM the scene is anchored to.'}
          </Text>
          {blocker ? (
            scene.product !== 'rDSM' && (
              <Text size="xs" c="dimmed">
                {blocker}
              </Text>
            )
          ) : (
            <Button size="xs" loading={running} onClick={() => void runAnchoring({ kind: 'terrain-tiles' })}>
              Anchor to DEM
            </Button>
          )}
          {anchor.sceneId === scene.id && anchor.status === 'error' && (
            <Text size="xs" c="red">
              {anchor.message}
            </Text>
          )}
        </Stack>
      </Alert>
    );
  }

  return (
    <Stack gap="md">
      <Text size="xs" c="dimmed" lh={1.5}>
        Water rises from a source and floods the connected lower ground first. Buildings are ranked by how early the water reaches them, how many people they likely hold and whether they house critical facilities.
      </Text>

      <Stack gap={6}>
        <Text size="sm" fw={500}>
          Where the water comes from
        </Text>
        <SegmentedControl
          size="xs"
          fullWidth
          value={u.floodSource}
          onChange={(v) => {
            u.set({ floodSource: v as FloodSourceKind, flood: null, floodStatus: 'idle', risks: [], playing: false, pickingSource: v === 'point' && !u.floodPoint });
          }}
          data={[
            { value: 'water', label: 'Water bodies' },
            { value: 'point', label: 'Chosen point' },
            { value: 'bathtub', label: 'Whole scene' },
          ]}
          aria-label="Flood source"
        />
        {u.floodSource === 'point' && (
          <Button size="xs" variant={u.pickingSource ? 'filled' : 'default'} leftSection={<IconCrosshair size={14} />} onClick={() => u.set({ pickingSource: !u.pickingSource })}>
            {u.pickingSource ? 'Click the terrain…' : u.floodPoint ? 'Move source point' : 'Choose source point'}
          </Button>
        )}
        {u.floodError && (
          <Text size="xs" c="orange" lh={1.5}>
            {u.floodError}
          </Text>
        )}
        {u.floodStatus === 'running' && (
          <Text size="xs" c="dimmed">
            Computing flood levels…
          </Text>
        )}
      </Stack>

      {flood && stats && (
        <>
          <Stack gap={6}>
            <Group justify="space-between">
              <Text size="sm" fw={500}>
                Water level
              </Text>
              <Text size="xs" className="dw-mono">
                +{rise.toFixed(1)} m · {(flood.baseLevel + rise).toFixed(1)} m a.s.l.
              </Text>
            </Group>
            <Group gap={8} wrap="nowrap">
              <Button size="compact-sm" variant="default" aria-label={u.playing ? 'Pause' : 'Play rising water'} onClick={() => u.set({ playing: !u.playing, rise: !u.playing && rise >= flood.maxRise - 0.05 ? 0 : rise })}>
                {u.playing ? <IconPlayerPause size={14} /> : <IconPlayerPlay size={14} />}
              </Button>
              <Slider style={{ flex: 1 }} min={0} max={flood.maxRise} step={0.1} value={rise} onChange={(v) => u.set({ rise: v, playing: false })} label={(v) => `+${v.toFixed(1)} m`} aria-label="Water level rise" />
            </Group>
            <Text size="xs" c="dimmed">
              Rise above the source’s normal level ({flood.baseLevel.toFixed(1)} m a.s.l.). Terrain spans {flood.minGround.toFixed(1)}–{flood.maxGround.toFixed(1)} m.
            </Text>
            <Switch size="xs" label="Show water on the terrain" checked={u.floodOverlay} onChange={(e) => u.set({ floodOverlay: e.currentTarget.checked })} />
            <Switch size="xs" label="Shade dry ground by how soon it floods" checked={u.vulnerability} onChange={(e) => u.set({ vulnerability: e.currentTarget.checked })} />
          </Stack>

          <Group gap="lg">
            <div>
              <Text size="xs" c="dimmed">
                Flooded
              </Text>
              <Text size="sm" fw={600} className="dw-mono">
                {area(stats.areaM2)} · {(stats.share * 100).toFixed(1)} %
              </Text>
            </div>
            <div>
              <Text size="xs" c="dimmed">
                Depth mean / max
              </Text>
              <Text size="sm" fw={600} className="dw-mono">
                {stats.meanDepthM.toFixed(1)} / {stats.maxDepthM.toFixed(1)} m
              </Text>
            </div>
          </Group>

          {!scene.objects?.buildings.length ? (
            <Text size="xs" c="dimmed" lh={1.5}>
              Building ranking needs the detected buildings (objects.json) from the model; this result has none. The water and vulnerability layers still work.
            </Text>
          ) : (
            <Stack gap={8}>
              <Group justify="space-between">
                <Text size="sm" fw={500}>
                  Emergency priority
                </Text>
                <Text size="xs" c="dimmed">
                  {wetNow} of {ranked.length} at-risk buildings under water now
                </Text>
              </Group>
              <Group gap={6}>
                {(['critical', 'high', 'watch'] as Tier[]).map((t) => (
                  <Badge key={t} size="sm" variant="light" style={{ background: `${TIER_COLORS[t]}22`, color: TIER_COLORS[t] }}>
                    {TIER_LABELS[t]} {tierCount[t]}
                  </Badge>
                ))}
              </Group>
              <Table withRowBorders={false} verticalSpacing={2} horizontalSpacing={4} fz="xs" highlightOnHover>
                <Table.Thead>
                  <Table.Tr>
                    <Table.Th>#</Table.Th>
                    <Table.Th>Priority</Table.Th>
                    <Table.Th ta="right">Floods at</Table.Th>
                    <Table.Th ta="right">Now</Table.Th>
                  </Table.Tr>
                </Table.Thead>
                <Table.Tbody>
                  {ranked.slice(0, 10).map((r, i) => {
                    const depth = buildingDepth(r, flood, rise);
                    return (
                      <Table.Tr key={r.index} style={{ cursor: 'pointer', background: u.focusBuilding === r.index ? 'var(--mantine-color-cyan-light)' : undefined }} onClick={() => u.set({ focusBuilding: u.focusBuilding === r.index ? null : r.index })}>
                        <Table.Td>{i + 1}</Table.Td>
                        <Table.Td>
                          <Tooltip label={`${r.areaM2.toFixed(0)} m² × ${r.storeys} storeys${r.facility ? ` · ${r.facility}` : ''}`} withArrow>
                            <span>
                              <span style={{ display: 'inline-block', width: 8, height: 8, background: TIER_COLORS[r.tier], marginRight: 6 }} />
                              {TIER_LABELS[r.tier]} · {r.score.toFixed(0)}
                              {r.facility ? ' ★' : ''}
                            </span>
                          </Tooltip>
                        </Table.Td>
                        <Table.Td ta="right" className="dw-mono">
                          +{r.rise.toFixed(1)} m
                        </Table.Td>
                        <Table.Td ta="right" className="dw-mono">
                          {depth > 0 ? `${depth.toFixed(1)} m deep` : 'dry'}
                        </Table.Td>
                      </Table.Tr>
                    );
                  })}
                </Table.Tbody>
              </Table>
              {ranked.length > 10 && (
                <Text size="xs" c="dimmed">
                  Top 10 of {ranked.length} buildings the water reaches. Click a row to outline it in 3D.
                </Text>
              )}
              <Text size="xs" c="dimmed" lh={1.5}>
                Score (0–100) = {WEIGHTS.urgency * 100} % urgency (how early it floods) + {WEIGHTS.occupancy * 100} % occupancy proxy (footprint × storeys, {STOREY_M} m each) + {WEIGHTS.critical * 100} % critical facility (★, from OpenStreetMap when the Facilities layer has loaded). Critical ≥ {TIER_LIMITS.critical}, High ≥ {TIER_LIMITS.high}.
              </Text>
            </Stack>
          )}
        </>
      )}

      <Text size="xs" c="dimmed" lh={1.5}>
        Indicative screening on a {scene.anchoring ? `${scene.anchoring.cellM.toFixed(0)} m` : '~30 m'} DEM ({scene.anchoring?.source ?? 'DEM'}): connected-ground bathtub, no rainfall, drainage, defences or flow speed. A higher-resolution DEM (Info tab → Load DEM file) sharpens it.
      </Text>
    </Stack>
  );
}
