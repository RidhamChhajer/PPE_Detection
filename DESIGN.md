---
version: alpha
name: AI Based PPE Monitoring & Warning frontend
description: Preserve the supplied UI_PPE monitor and extend its operator pages.
colors:
  primary: "#00A854"
  background: "#F1F5F9"
  surface: "#FFFFFF"
  ink: "#0F172A"
  danger: "#EF4444"
  blue: "#0284C7"
spacing:
  panel: "24px"
  gap: "14px"
rounded:
  card: "22px"
  control: "12px"
typography:
  body:
    fontFamily: "Plus Jakarta Sans"
---

# Design authority

The original frontend/src/App.css, frontend/src/index.css and frontend/src/components/icons.jsx own the supplied visual identity. Keep their main-screen proportions, colours, typography, PPE illustrations, status pills, rounded panels and Bekaert footer. Runtime ownership follows the existing CSS system: index.css semantic variables → App.css reference components / Workspace.css operator variants. DESIGN.md mirrors this system; it does not generate another token layer. The approved project name is AI Based PPE Monitoring & Warning; use the user-supplied goggles and Bekaert images unchanged.

The audience is factory workers and operators using a portrait display in India; English UI, en-IN/en-GB dates and Asia/Kolkata time. Japan-specific flows do not apply. Fonts are the original Plus Jakarta Sans and Inter, bundled in frontend/public/fonts with their licenses for offline use. Body, labels and headings retain the original font stack. Existing fixed values in App.css remain reference-owned variants.

Additions: a collapsible navigation rail, Test media, synthetic sample dashboard and Settings. Sidebar owns navigation; reusable ui-button, form-field, ui-notice and native select own operator controls. Native popup geometry is accepted. The approved main screen is inside its existing kiosk wrapper. Production assets are emitted under /static by frontend/vite.config.js. Font CSS resolves bundled files relative to its location; brand styles are external CSS to preserve the self-only Content Security Policy. Live uses the full remaining width beside the rail, with no 640px desktop cap or centering gutters; camera panel geometry is independent of capture resolution and frame proportions. Canvas previews fill fixed responsive panels with object-fit: contain; borders are accepted so the whole frame remains visible. Waiting/retry states use the same panel geometry. The larger upper-camera panel (16:10) stays above the smaller lower-camera panel (16:6) at every viewport width; browser zoom and small screens use natural document scrolling. The PPE status row contains goggles, clear wrists, shoes and earplugs. Earplugs use a corded-plug outline and a neutral, permanent In progress status; no earplug detector is connected. Four columns become two on narrow screens. The user-selected wrist symbol is option C: a crossed-out watch outline, reused in the Clear Wrists status card and the upper-camera accessory strip. It represents the wrist-accessory rule; detection still includes all trained wrist accessories.

Live uses actual camera streams and real engine status. It never displays sample photos, figures or simulated bounding boxes. The bottom connection and fullscreen sidebar icons are removed. Only the Sample dashboard uses synthetic data. No audio alert, fake latency, simulated production detections or worker identity is carried over. Green/red show positive/violation; gray shows unavailable evidence. Online text must match current connection. All new actions have 44px targets, visible keyboard focus, inline errors, bounded loading, and reduced-motion support. All scroll surfaces inherit a visible scrollbar baseline; forced colors defer to the system.

## Canonical UI map

| Capability | Owner | Variants | Verification |
|---|---|---|---|
| Navigation | App.jsx sidebar | collapsed, expanded, narrow overlay | tests/browser_check.py |
| Button / feedback | Workspace.css ui-button / ui-notice / ui-toast | primary, disabled, busy, error | browser evidence |
| Form / select | OperatorPages.jsx, native labelled controls | settings, media | browser evidence |
| Unsaved draft dialog | App.jsx native dialog | stay, discard, Escape | browser evidence |
| Scrollbar | Workspace.css global baseline | normal, forced colors | browser evidence |

Keep new controls visually consistent across pages. Do not replace the reference layout with new cards or a new theme. Do not copy the main project's previously rejected frontend into this directory.

The separate-preview connection/disconnect settings card is retired. Four metric cards are placeholders for total workers detected, violations, violation percentage, and workers currently in Camera 1. Each shows an em dash and Coming soon; no counting logic is connected. GPU identity and per-model inference timings and rates appear under Settings > Inference & hardware. Preview/detection rates remain in each video caption. Camera source names and shared-camera text explain the current feed selection.

Camera previews retain the source canvas dimensions and fit the complete image inside stable panels with finite latest-frame requests instead of MJPEG images. No-frame and retry states preserve the existing empty-panel visual language; GPU staleness does not remount the preview.

Branding update: the supplied goggles bitmap replaces the prior outline in both goggles locations. The compact footer has the user-supplied VIT Pune logo on the left, a central project-team list, and the supplied Bekaert image on the right. Full names: Ridham Chhajer, Kedar Naygaonkar, Kedar Pethkar, Sanskruti Chaure, Sahil Nehate, Om Parmar. Original image pixels and aspect ratios are preserved; CSS provides responsive display sizes. Logos remain local/offline assets.

PPE status cards use whole-card backgrounds: positive #166534 and violation #991b1b, with white result text. White icon circles and label pills retain legible dark artwork/text. Unknown and pending cards remain neutral. The supplied goggles artwork is unchanged. Status words and symbols remain present so color is not the sole signal.

The footer is a shallow, full-width rectangle; logos keep their original aspect ratios. On mobile the two logos share a row above the team names. The header uses the title AI Based PPE Monitoring & Warning and the exact tagline Safety First,No Harm to ANYONE. Date/time, system status, and Start/Stop detection remain in one row in that order; GPU hardware text belongs only in Settings.

Repository cleanup removes retired kiosk simulation controls, mock bounding boxes, diagnostic overlays, and unused animations. Active kiosk styles remain in App.css; Workspace.css owns the global scrollbar baseline. Source assets and generated app/static assets are both retained for development and offline operation.

Camera layout: source resolution changes only canvas backing pixels, never panel geometry. Native per-camera fullscreen expands the existing preview and preserves overlays; the exit control or Escape restores the dashboard. Ctrl +/- uses ordinary browser zoom and responsive breakpoints, with scrolling when space is limited. No capture or model settings change with the display layout.
