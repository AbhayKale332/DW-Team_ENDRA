import { useEffect, useRef } from 'react';
import { ActionIcon, Group, ScrollArea, SegmentedControl, Stack, Text } from '@mantine/core';
import { IconX } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useUseCases, type UseCase } from '@/store/usecases';
import { usePoi, loadPois } from '@/features/poi/poiStore';
import { useView } from '@/store/view';
import { canGeolocate } from '@/lib/osm';
import { invalidateUseCases, refreshCoverage, rerankWithFacilities } from './actions';
import { TelecomPanel } from './TelecomPanel';
import { FloodPanel } from './FloodPanel';

const TITLES: Record<UseCase, string> = { telecom: 'Telecom tower coverage', flood: 'Flood response planning' };

/** Scenarios: what the generated DSM is good for. A floating panel over the viewport, opened from the header. */
export function ScenarioOverlay() {
  const scene = useScene((s) => s.scene);
  const open = useUseCases((s) => s.open);
  const active = useUseCases((s) => s.active);
  const pois = usePoi((s) => s.status);
  const lastScene = useRef<unknown>(null);

  // A new scene object (new result, anchoring finished, height reference switched) invalidates results.
  useEffect(() => {
    if (lastScene.current === scene) return;
    const first = lastScene.current === null;
    lastScene.current = scene;
    if (first) return;
    invalidateUseCases();
    if (scene && useUseCases.getState().towers.length) refreshCoverage(0);
  }, [scene]);

  // Facilities raise the priority of hospitals and schools: fetch them (georeferenced scenes only), then re-rank.
  useEffect(() => {
    if (!open || active !== 'flood' || !scene || !canGeolocate(scene.georef) || !useView.getState().poi) return;
    void loadPois(scene);
  }, [open, active, scene]);
  useEffect(() => {
    if (pois === 'ready') rerankWithFacilities();
  }, [pois]);

  if (!open || !scene) return null;
  const close = () => useUseCases.getState().set({ open: false, placing: false, pickingSource: false, playing: false });
  return (
    <div
      className="dw-float"
      role="dialog"
      aria-label="Scenarios"
      style={{ position: 'absolute', top: 162, left: 14, zIndex: 4, width: 356, maxHeight: 'calc(100% - 232px)', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}
    >
      <Group justify="space-between" wrap="nowrap" px="sm" pt="xs" pb={6}>
        <Text size="sm" fw={600}>
          {TITLES[active]}
        </Text>
        <ActionIcon size="sm" variant="subtle" color="gray" onClick={close} aria-label="Close scenarios">
          <IconX size={15} />
        </ActionIcon>
      </Group>
      <Stack gap="sm" px="sm" pb="xs">
        <SegmentedControl
          size="xs"
          fullWidth
          value={active}
          onChange={(v) => useUseCases.getState().set({ active: v as UseCase, placing: false, pickingSource: false, playing: false })}
          data={[
            { value: 'telecom', label: 'Telecom' },
            { value: 'flood', label: 'Flood' },
          ]}
          aria-label="Scenario"
        />
      </Stack>
      <ScrollArea style={{ flex: 1 }} type="auto" offsetScrollbars>
        <Stack px="sm" pb="sm">
          {active === 'telecom' ? <TelecomPanel /> : <FloodPanel />}
        </Stack>
      </ScrollArea>
    </div>
  );
}
