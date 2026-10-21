---
version: alpha
name: PPE Detection
description: Local factory video review and simultaneous face and feet monitoring.
colors:
  primary: "#176754"
  background: "#f3f5f7"
  surface: "#ffffff"
  text: "#172c3b"
typography:
  body:
    fontFamily: "Segoe UI, Arial, sans-serif"
rounded:
  control: "8px"
  card: "14px"
spacing:
  panel-gap: "24px"
components:
  button: {}
  card: {}
  camera-select: {}
---

# PPE Detection design

## Overview

A desktop instrument panel for factory operators reviewing visible PPE. Preserve
the established quiet white cards and green actions; the paired, labeled camera
views identify the two inspection zones. English interface for the user's factory
workflow. No Japan-specific data or localization. Product register, not marketing.
Product behavior is documented in MD DOC/CAMERA_INTEGRATION.md.

## Colors

Runtime CSS in app/static/style.css owns tokens; this document mirrors them.
The four color tokens map directly to --color-primary, --color-background,
--color-surface and --color-text. Green means an action or visible detection;
it does not certify compliance. Errors also have text and alert semantics.

## Typography

The body token maps to --font-body. Native Windows Segoe UI keeps controls familiar.
Headings use size and weight in the same family; technical frame data stays readable.

## Layout

Maximum content width is 1200px. Existing upload controls and output remain side
by side; live controls sit above two equal camera cards. At 850px and below,
panels stack in face/feet order. Document scrolling remains available. Camera
media reserves a 4:3 region and contains the actual image without cropping.

## Elevation & Depth

White cards use a border and the existing subtle shadow. Avoid extra overlays.

## Shapes

Control/card radii map to --radius-control and --radius-card. Panels are separated
by the 24px gap used by both upload and live layouts.

## Components

Native selects own camera popup behavior and keyboard interaction. Labels and
inline errors explain assignments. Either role can be Not used; a shared camera
produces one combined preview, while two different cameras stay role-specific.
Existing CSS owns buttons, cards, badges and visible focus. Pending
actions disable repeat submissions. One paired start/stop action controls live
capture, and each preview displays its own state. Upload retains both detectors.
The selected inference backend/device is visible with expandable fallback reasons.
Tabs support arrow keys, Home and End. Status uses live regions; streaming frame
metrics do not announce every update. Polling is sequential and previews are
hidden on failure or stale data. Scrollbars use global CSS tokens and system
colors in forced-color mode. Reduced motion disables the dropzone transition.

## Do's and Don'ts

- Keep the upload workflow and camera controls consistent in wording and feedback.
- Keep errors actionable and visible near the relevant controls.
- Do not use missing detections as evidence of non-compliance.
- Do not introduce a second UI framework or a new visual identity for one feature.
