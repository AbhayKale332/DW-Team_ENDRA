import { Group, Stack, Table, Text, Tooltip } from '@mantine/core';
import { IconInfoCircle } from '@tabler/icons-react';

/** Shared building blocks for the docked panels and the scenario card, so every section reads the same. */

/** A titled block: uppercase section title, optional right-side badge/actions, then the content. */
export function PanelSection({ title, right, hint, children, gap = 'sm' }: { title: string; right?: React.ReactNode; hint?: React.ReactNode; children: React.ReactNode; gap?: string | number }) {
  return (
    <Stack gap={gap} component="section" aria-label={title}>
      <Group justify="space-between" gap="xs" wrap="nowrap" mih={20}>
        <Group gap={4} wrap="nowrap">
          <span className="dw-section-title">{title}</span>
          {hint && <InfoHint>{hint}</InfoHint>}
        </Group>
        {right}
      </Group>
      {children}
    </Stack>
  );
}

/** Label / value rows: dimmed label on the left, monospaced value on the right. */
export function KeyValueRows({ rows }: { rows: Array<[string, React.ReactNode]> }) {
  return (
    <Table withRowBorders={false} verticalSpacing={3} horizontalSpacing={0} fz="xs">
      <Table.Tbody>
        {rows.map(([k, v]) => (
          <Table.Tr key={k}>
            <Table.Td c="dimmed" w="44%" valign="top">
              {k}
            </Table.Td>
            <Table.Td className="dw-mono" ta="right" style={{ overflowWrap: 'anywhere' }}>
              {v}
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}

/** One empty state for panels that need a scene (or something else) first. */
export function EmptyPanel({ icon: Icon, children }: { icon: typeof IconInfoCircle; children: React.ReactNode }) {
  return (
    <Stack align="center" gap={8} py="xl" px="md">
      <Icon size={28} stroke={1.4} color="var(--dw-faint)" aria-hidden />
      <Text size="sm" c="dimmed" ta="center">
        {children}
      </Text>
    </Stack>
  );
}

/** A small (i) that holds method notes and caveats, instead of paragraphs on screen. */
export function InfoHint({ children, label = 'More information' }: { children: React.ReactNode; label?: string }) {
  return (
    <Tooltip label={children} multiline maw={300} withArrow events={{ hover: true, focus: true, touch: true }}>
      <span tabIndex={0} role="img" aria-label={label} style={{ display: 'inline-flex', color: 'var(--dw-faint)', cursor: 'help' }}>
        <IconInfoCircle size={14} stroke={1.6} />
      </span>
    </Tooltip>
  );
}
