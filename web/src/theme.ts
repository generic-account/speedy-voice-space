// Central dark-mode palette. Elevation reads as lighter surfaces on a near-black
// page (not white), with subtle gray borders. Referenced from JSX styles and the
// canvas drawing code; the matching page/form-control colors live in index.html.
export const C = {
  bg: "#000000", // page background (true black)
  surface: "#1c1c1c", // raised cards/panels
  canvas: "#141414", // plot + strip backgrounds
  border: "#3a3a3a", // card/control borders
  text: "#ffffff", // primary text + titles (true white)
  label: "#a0a0a0", // secondary labels
  muted: "#8a8a8a", // axis ticks, status line
  grid: "#2a2a2a", // gridlines
  accent: "#5aa6ff", // links, current point
  f2: "#4a9eff", // F2 strip line
  f3: "#e0843c", // F3 strip line
  error: "#ef6b6b",
  trailRGB: "230,230,230", // trail dots (used in rgba())
};
