import { useEffect, useRef } from 'react';
import { useComputedColorScheme } from '@mantine/core';
import * as echarts from 'echarts/core';
import { BarChart, HeatmapChart, LineChart } from 'echarts/charts';
import { AriaComponent, GridComponent, MarkLineComponent, TooltipComponent, VisualMapComponent, LegendComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsCoreOption, ECharts } from 'echarts/core';

echarts.use([LineChart, BarChart, HeatmapChart, GridComponent, TooltipComponent, VisualMapComponent, MarkLineComponent, AriaComponent, LegendComponent, CanvasRenderer]);

export interface ChartTheme {
  ink: string;
  dim: string;
  line: string;
  bg: string;
}

export function chartTheme(scheme: 'light' | 'dark'): ChartTheme {
  return scheme === 'dark'
    ? { ink: '#e8ecf1', dim: '#8a949e', line: '#252c34', bg: 'transparent' }
    : { ink: '#151a20', dim: '#5b6572', line: '#dde2e8', bg: 'transparent' };
}

export function baseAxis(t: ChartTheme, name: string) {
  return {
    name,
    nameLocation: 'middle' as const,
    nameGap: 26,
    nameTextStyle: { color: t.dim, fontSize: 10 },
    axisLine: { lineStyle: { color: t.line } },
    axisTick: { lineStyle: { color: t.line } },
    axisLabel: { color: t.dim, fontSize: 9,fontFamily: 'JetBrains Mono, monospace' },
    splitLine: { lineStyle: { color: t.line, opacity: 0.6 } },
  };
}

/** Thin ECharts wrapper: theme-aware, resizes with its container, exposes the instance for PNG export. */
export function EChart({
  option,
  height,
  ariaLabel,
  onReady,
  onEvents,
}: {
  option: (t: ChartTheme) => EChartsCoreOption;
  height: number;
  ariaLabel: string;
  onReady?: (c: ECharts) => void;
  onEvents?: Record<string, (p: unknown) => void>;
}) {
  const el = useRef<HTMLDivElement>(null);
  const chart = useRef<ECharts | null>(null);
  const scheme = useComputedColorScheme('light');

  useEffect(() => {
    if (!el.current) return;
    const c = echarts.init(el.current, undefined, { renderer: 'canvas' });
    chart.current = c;
    onReady?.(c);
    const ro = new ResizeObserver(() => c.resize());
    ro.observe(el.current);
    return () => {
      ro.disconnect();
      c.dispose();
      chart.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const c = chart.current;
    if (!c) return;
    const t = chartTheme(scheme);
    c.setOption({ aria: { enabled: true, label: { description: ariaLabel } }, textStyle: { fontFamily: 'Inter, sans-serif' }, ...option(t) }, true);
  }, [option, scheme, ariaLabel]);

  useEffect(() => {
    const c = chart.current;
    if (!c || !onEvents) return;
    for (const [k, fn] of Object.entries(onEvents)) c.on(k, fn);
    return () => {
      for (const [k, fn] of Object.entries(onEvents)) c.off(k, fn);
    };
  }, [onEvents]);

  return <div ref={el} style={{ width: '100%', height }} role="img" aria-label={ariaLabel} />;
}
