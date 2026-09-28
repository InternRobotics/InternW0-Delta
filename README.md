# InternW0-Δ project website

**An Embodied World Model Bridging Predictive Dynamics and Actions**

This branch contains only the static research website and its approved media.
Model code is maintained separately on the repository's main development branch.

## GitHub Pages

Intended publishing source: **Deploy from a branch → gh-pages → /(root)**.
The `.nojekyll` file disables Jekyll processing. No package installation or build
step is needed to serve this branch.

Expected website address after deployment:
https://internrobotics.github.io/InternW0-Delta/

## Editing

- `content.js`: resource links, demonstrations, figures and benchmark values.
- `index.html`: page structure and the no-JavaScript benchmark table.
- `translations.js`: Chinese translations.
- `styles.css` and `themes/studio-title.css`: the selected mint visual design.
- `inline-video.js`: visibility-driven inline playback.
- `assets/`: the website's approved logos, figures, posters and videos.

The website uses relative asset paths so it can be served at the project's
GitHub Pages subpath. Videos play muted when visible and pause offscreen.
Native controls remain available when browser settings prevent autoplay.

Benchmark values on the webpage may be newer than values embedded in the
project film; updating a webpage value does not modify the video.

Third-party mark sources are recorded in `assets/marks/SOURCES.md`.
