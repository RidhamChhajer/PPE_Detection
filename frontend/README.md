# AI Based PPE Monitoring & Warning React frontend

This is the maintained frontend source. Production runs through the existing Python launcher at http://127.0.0.1:8876; Node.js is not required on the factory PC.

For development, run `npm ci` once here, then `npm run dev` and open http://127.0.0.1:5174/static/. The Vite proxy forwards API requests to the Python service. Run `npm run lint` and `npm run build` after edits. Build replaces app/static with the compiled production frontend. The output directory is dedicated to generated assets; do not put source files there.

The project-root DESIGN.md and UX-CONTRACT.md own the maintained design/interaction decisions. Fonts and their licenses live in public/fonts; no external assets are required. Live streams and media testing use the same-origin /api routes. Source sample photos and unused starter assets are not included.
