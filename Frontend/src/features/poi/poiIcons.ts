import { createElement } from 'react';
import * as THREE from 'three';
import {
  IconAmbulance,
  IconBabyCarriage,
  IconBuildingBank,
  IconBuildingHospital,
  IconBus,
  IconCertificate,
  IconFirstAidKit,
  IconFlame,
  IconHelicopterLanding,
  IconPill,
  IconPlane,
  IconSchool,
  IconShieldCheck,
  IconStethoscope,
  IconTent,
  IconTrain,
  IconUsersGroup,
} from '@tabler/icons-react';
import { POI_KIND_META, POI_KINDS, type PoiCategory, type PoiKind } from '@/lib/poi';

export const POI_KIND_ICONS: Record<PoiKind, typeof IconFlame> = {
  hospital: IconBuildingHospital,
  ambulance_station: IconAmbulance,
  fire_station: IconFlame,
  police: IconShieldCheck,
  clinic: IconStethoscope,
  school: IconSchool,
  university: IconCertificate,
  kindergarten: IconBabyCarriage,
  shelter: IconTent,
  townhall: IconBuildingBank,
  community_centre: IconUsersGroup,
  pharmacy: IconPill,
  rail_station: IconTrain,
  bus_station: IconBus,
  helipad: IconHelicopterLanding,
  aerodrome: IconPlane,
};

export const POI_CATEGORY_ICONS: Record<PoiCategory, typeof IconFlame> = {
  emergency: IconFirstAidKit,
  education: IconSchool,
  civic: IconTent,
  transport: IconTrain,
};

/** Atlas layout: one CELL-px square per kind, ATLAS_COLS × ATLAS_COLS cells (≥ the number of kinds), in POI_KINDS order. */
export const ATLAS_COLS = 4;
const CELL = 64;

let atlas: Promise<THREE.CanvasTexture> | null = null;

/** Every kind's map pin (coloured disc + white icon) in one texture, built once. react-dom/server is only
 *  loaded here, so it stays out of the main bundle. */
export function poiAtlas(): Promise<THREE.CanvasTexture> {
  atlas ??= (async () => {
    const { renderToStaticMarkup } = await import('react-dom/server');
    const canvas = document.createElement('canvas');
    canvas.width = ATLAS_COLS * CELL;
    canvas.height = ATLAS_COLS * CELL;
    const ctx = canvas.getContext('2d')!;
    await Promise.all(
      POI_KINDS.map(async (kind, i) => {
        const x = (i % ATLAS_COLS) * CELL;
        const y = Math.floor(i / ATLAS_COLS) * CELL;
        ctx.save();
        ctx.shadowColor = 'rgba(0,0,0,0.45)';
        ctx.shadowBlur = 5;
        ctx.beginPath();
        ctx.arc(x + CELL / 2, y + CELL / 2, CELL / 2 - 6, 0, Math.PI * 2);
        ctx.fillStyle = POI_KIND_META[kind].color;
        ctx.fill();
        ctx.restore();
        ctx.lineWidth = 3;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
        const svg = renderToStaticMarkup(createElement(POI_KIND_ICONS[kind], { size: 34, color: '#ffffff', stroke: 2 }));
        const img = new Image();
        img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
        await img.decode();
        ctx.drawImage(img, x + (CELL - 34) / 2, y + (CELL - 34) / 2, 34, 34);
      }),
    );
    const tex = new THREE.CanvasTexture(canvas);
    tex.flipY = false; // cell rows top-down, like gl_PointCoord
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.generateMipmaps = false;
    tex.minFilter = THREE.LinearFilter;
    return tex;
  })();
  atlas.catch(() => (atlas = null)); // let a later mount try again
  return atlas;
}
