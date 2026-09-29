import { useEffect, useRef } from 'react';
import { ActionIcon, Group, ScrollArea, SegmentedControl, Text } from '@mantine/core';
import { IconAntenna, IconDroplet, IconX } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useUseCases, type UseCase } from '@/store/usecases';
import { usePoi, loadPois } from '@/features/poi/poiStore';
import { useView } from '@/store/view';
import { canGeolocate } from '@/lib/osm';
import { invalidateUseCases, refreshCoverage, rerankWithFacilities } from './actions';
import { TelecomPanel } from './TelecomPanel';
import { FloodPanel } from './FloodPanel';
import classes from '@/features/viewport/overlays/overlays.module.css';

const TITLES: Record<UseCase, string> = { telecom: 'Telecom tower coverage', flood: 'Flood response planning' };

function Tab({ icon: Icon, label }: { icon: typeof IconAntenna; label: string }) {
  return (
    <Group gap={6} justify="center" wrap="nowrap">
      <Icon size={14} stroke={1.7} aria-hidden />
      <span>{label}</span>
    </Group>
  );
}

/** Scenarios: what the generated DSM is good for. A card in the viewport's top-left column, opened from the header. */
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
    <div className={`dw-float ${classes.scenarioCard}`} role="dialog" aria-label="Scenarios">
      <Group justify="space-between" wrap="nowrap" px="sm" pt="sm" pb="xs" style={{ borderBottom: '1px solid var(--dw-line)' }}>
        <Text size="sm" fw={600}>
          {TITLES[active]}
        </Text>
        <ActionIcon size="sm" variant="subtle" color="gray" onClick={close} aria-label="Close scenarios">
          <IconX size={15} />
        </ActionIcon>
      </Group>
      <div style={{ padding: 'var(--dw-gap-3) var(--dw-gap-3) 0' }}>
        <SegmentedControl
          size="xs"
          fullWidth
          value={active}
          onChange={(v) => useUseCases.getState().set({ active: v as UseCase, placing: false, pickingSource: false, playing: false })}
          data={[
            { value: 'telecom', label: <Tab icon={IconAntenna} label="Telecom" /> },
            { value: 'flood', label: <Tab icon={IconDroplet} label="Flood" /> },
          ]}
          aria-label="Scenario"
        />
      </div>
      <ScrollArea type="auto" offsetScrollbars style={{ flex: '1 1 auto', minHeight: 0 }}>
        <div style={{ padding: 'var(--dw-gap-3)' }}>{active === 'telecom' ? <TelecomPanel /> : <FloodPanel />}</div>
      </ScrollArea>
    </div>
  );
}
