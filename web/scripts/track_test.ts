// Port-correctness check: run the TS FormantTracker on the experiment candidate
// CSVs and report jump counts (compare to the Python tracker numbers).
//   node scripts/track_test.ts
import fs from "node:fs";
import { FormantTracker, type Pole } from "../src/processing/formantTracker.ts";

function jumps(a: (number | null)[]): number {
  const v = a.filter((x): x is number => x !== null);
  let n = 0;
  for (let i = 1; i < v.length; i++) if (Math.abs(v[i] - v[i - 1]) > 300) n++;
  return n;
}

const DATA = new URL("../../tools/experiments/data/", import.meta.url);
for (const stem of ["rec_9", "rec_10", "rec_11_aoiue", "rec_12_aeoua", "rec_u5"]) {
  const txt = fs.readFileSync(new URL(`${stem}.csv`, DATA), "utf8").trim().split("\n");
  const header = txt[0].split(",");
  const col = (name: string) => header.indexOf(name);
  const tr = new FormantTracker();
  const f2: (number | null)[] = [];
  const f3: (number | null)[] = [];
  for (let r = 1; r < txt.length; r++) {
    const cells = txt[r].split(",");
    if (cells[col("voiced")] !== "1") continue;
    const cand: Pole[] = [];
    for (let c = 1; c <= 6; c++) {
      const f = cells[col(`c${c}f`)];
      const b = cells[col(`c${c}b`)];
      if (f) cand.push([Number(f), b ? Number(b) : 500]);
    }
    const t = tr.push(cand);
    if (t) {
      f2.push(t[1]);
      f3.push(t[2]);
    }
  }
  console.log(`${stem}: F2 jumps=${jumps(f2)}  F3 jumps=${jumps(f3)}`);
}
