// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import mermaid from 'astro-mermaid';
import starlightLinksValidator from 'starlight-links-validator';

export default defineConfig({
	site: 'https://depthwizard-docs.vercel.app',
	devToolbar: { enabled: false },
	integrations: [
		mermaid({
			theme: 'neutral',
			autoTheme: true,
			enableLog: false,
			mermaidConfig: {
				fontFamily: 'Inter, system-ui, sans-serif',
				flowchart: { curve: 'basis', padding: 12, nodeSpacing: 36, rankSpacing: 44 },
				sequence: { mirrorActors: false, actorMargin: 28, boxMargin: 6, messageMargin: 28, useMaxWidth: true },
			},
		}),
		starlight({
			title: 'DepthWizard',
			description:
				'Single-view metric height estimation and 3D flythrough from aerial and satellite imagery.',
			logo: { src: './src/assets/logo.svg', alt: 'DepthWizard' },
			favicon: '/favicon.svg',
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/AbhayKale332/DepthWizard' },
			],
			editLink: {
				baseUrl: 'https://github.com/AbhayKale332/DepthWizard/edit/main/Docs-Site/',
			},
			customCss: [
				'@fontsource/inter/400.css',
				'@fontsource/inter/500.css',
				'@fontsource/inter/600.css',
				'@fontsource/inter/700.css',
				'@fontsource/jetbrains-mono/400.css',
				'katex/dist/katex.min.css',
				'./src/styles/theme.css',
			],
			tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 3 },
			lastUpdated: false,
			plugins: [starlightLinksValidator()],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', link: '/' },
						'overview/motivation',
						'overview/quickstart',
						'overview/concepts',
					],
				},
				{
					label: 'Model Architecture',
					items: [
						'model/overview',
						'model/encoder',
						'model/decoder-heads',
						'model/losses',
						'model/inference',
					],
				},
				{
					label: 'Training & Fine-tuning',
					items: [
						'training/datasets',
						'training/preprocessing',
						'training/recipe',
						'training/versions',
						'training/findings',
					],
				},
				{
					label: 'Evaluation & Results',
					items: [
						'results/metrics',
						'results/benchmarks',
						'results/landscape-accuracy',
						'results/error-analysis',
						'results/lidar-validation',
						'results/gallery',
					],
				},
				{
					label: 'Georeferencing & Scale',
					items: ['geo/scale-recovery', 'geo/absolute-dsm', 'geo/dem-sources'],
				},
				{
					label: 'System Architecture',
					items: [
						'system/overview',
						'system/hosted-path',
						'system/self-hosted',
						'system/api-reference',
						'system/outputs',
						'system/onnx',
						'system/frontend-architecture',
						'system/deployment',
					],
				},
				{
					label: 'User Guide',
					items: [
						'guide/interface',
						'guide/first-estimate',
						'guide/navigation',
						'guide/layers',
						'guide/validation',
						'guide/anchoring',
						'guide/scenarios',
						'guide/export',
						'guide/shortcuts',
						'guide/troubleshooting',
					],
				},
				{
					label: 'Reference',
					items: [
						'reference/limitations',
						'reference/paper',
						'reference/changelog',
						'reference/team',
					],
				},
			],
		}),
	],
});
