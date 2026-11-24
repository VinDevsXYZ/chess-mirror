// Personal Chess Mirror — dashboard front-end (vanilla JS, no build step, fully offline).
// All text is inserted with textContent / DOM nodes (see h()), never as raw HTML.

// ================================================================== DOM helpers
const SVG_NS = "http://www.w3.org/2000/svg";

function applyAttrs(el, attrs) {
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value == null || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value === true ? "" : String(value));
  }
}

function appendChildren(el, children) {
  for (const child of children.flat(Infinity)) {
    if (child == null || child === false || child === "") continue;
    el.append(child instanceof Node ? child : String(child));
  }
  return el;
}

/** Create an HTML element: h("p", {class: "x"}, "text", childNode, [more]) */
function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  applyAttrs(el, attrs);
  return appendChildren(el, children);
}

/** Create an SVG element. */
function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  applyAttrs(el, attrs);
  return appendChildren(el, children);
}

// ================================================================== formatting
const num = (x) => (x === null || x === undefined || x === "" ? NaN : Number(x));
const ok = (x) => Number.isFinite(num(x));
const fmt = (x, digits = 0) =>
  ok(x) ? num(x).toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits }) : "–";
const pct = (x, digits = 0) => (ok(x) ? `${(num(x) * 100).toFixed(digits)}%` : "–");
const pctRange = (ci, digits = 0) =>
  Array.isArray(ci) && ok(ci[0]) && ok(ci[1]) ? `${(ci[0] * 100).toFixed(digits)}–${(ci[1] * 100).toFixed(digits)}%` : "";
const range = (lo, hi, digits = 0) => (ok(lo) && ok(hi) ? `${fmt(lo, digits)}–${fmt(hi, digits)}` : "");
const signed = (x, digits = 2) =>
  ok(x) ? `${num(x) > 0 ? "+" : num(x) < 0 ? "−" : "±"}${Math.abs(num(x)).toFixed(digits)}` : "–";
const capitalize = (s) => (s ? String(s)[0].toUpperCase() + String(s).slice(1) : "");
const plural = (n, word, many = `${word}s`) => `${fmt(n)} ${num(n) === 1 ? word : many}`;
const poolName = (pool) => String(pool || "").replace(":", " · ");

function probText(p) {
  if (!ok(p)) return "–";
  if (p >= 0.9995) return "> 99.9%";
  if (p <= 0.0005) return "< 0.1%";
  return pct(p, p > 0.99 || p < 0.01 ? 1 : 0);
}

function parseDate(s) {
  if (!s) return null;
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(s);
  const d = m ? new Date(+m[1], +m[2] - 1, +m[3]) : new Date(s);
  return Number.isNaN(d.getTime()) ? null : d;
}

function fmtDate(s) {
  const d = parseDate(s);
  return d ? d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" }) : s ? String(s) : "–";
}

function fmtClock(seconds) {
  if (!ok(seconds)) return "–";
  const s = Math.round(num(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** "e4 e5 Nf3" -> "1.e4 e5 2.Nf3" */
function numberLine(sans) {
  const list = Array.isArray(sans) ? sans : String(sans || "").split(/\s+/).filter(Boolean);
  return list.map((san, i) => (i % 2 === 0 ? `${i / 2 + 1}.${san}` : san)).join(" ");
}

// ================================================================== evidence tiers
const FACT = "observed fact";
const INFERENCE = "statistical inference";
const HYPOTHESIS = "weak hypothesis";
const INSUFFICIENT = "insufficient evidence";

const TIERS = {
  [FACT]: { cls: "fact", help: "Raw counts from your games." },
  [INFERENCE]: { cls: "inference", help: "At least 95% probability of a real difference, with 30+ opportunities." },
  [HYPOTHESIS]: { cls: "hypothesis", help: "Probably real (75%+), or based on few observations. Treat with caution." },
  [INSUFFICIENT]: { cls: "insufficient", help: "Not enough data to call this a tendency." },
};

const isInsufficient = (tier) => !tier || !TIERS[tier] || tier === INSUFFICIENT;

function tierBadge(tier) {
  if (!tier) return null;
  const known = TIERS[tier];
  return h("span", { class: `badge tier-${known ? known.cls : "insufficient"}`, title: known ? known.help : tier }, tier);
}

function tierLegend() {
  return h("div", { class: "legend" },
    h("span", null, "Evidence:"),
    Object.entries(TIERS).map(([name, t]) => h("span", { class: "item", title: t.help }, tierBadge(name), h("span", null, t.help))));
}

// ================================================================== API
async function api(path, { method = "GET", body, refresh = false } = {}) {
  const url = refresh ? `${path}${path.includes("?") ? "&" : "?"}refresh=1` : path;
  const init = { method };
  if (body !== undefined) {
    init.headers = { "Content-Type": "application/json" };
    init.body = JSON.stringify(body);
  }
  const res = await fetch(url, init);
  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (!res.ok) throw new Error((data && data.error) || `${res.status} ${res.statusText}`);
  return data;
}

/** Memoise a loader so several sections can share one request; failures are not cached. */
function shared(loader) {
  let promise = null;
  return () => {
    if (!promise) promise = loader().catch((err) => { promise = null; throw err; });
    return promise;
  };
}

// ================================================================== generic UI pieces
function loadingEl(text = "Loading…") {
  return h("div", { class: "loading" }, h("span", { class: "spinner", "aria-hidden": "true" }), h("span", null, text));
}

function errorBox(err, retry) {
  return h("div", { class: "error-box", role: "alert" },
    h("span", null, `Could not load this: ${err && err.message ? err.message : err}`),
    retry && h("button", { class: "btn btn-small", type: "button", onclick: retry }, "Try again"));
}

const emptyNote = (text) => h("p", { class: "empty" }, text);

/** A container that shows loading → render(data) | error with retry. Render errors are contained too. */
function asyncBlock(loader, render, loadingText) {
  const box = h("div", { class: "async" });
  const run = () => {
    box.replaceChildren(loadingEl(loadingText));
    Promise.resolve()
      .then(loader)
      .then((data) => box.replaceChildren(...[render(data)].flat(Infinity).filter(Boolean)))
      .catch((err) => {
        console.warn(err);
        box.replaceChildren(errorBox(err, run));
      });
  };
  run();
  return box;
}

function pageHead(title, text) {
  return h("div", { class: "page-head" }, h("h1", null, title), text && h("p", null, text));
}

function card({ title, sub, action, cls = "" } = {}, ...children) {
  const head = title || action
    ? h("div", { class: "card-head" }, h("div", { class: "row" }, title && h("h2", null, title), sub && h("span", { class: "sub" }, sub)), action)
    : null;
  return h("section", { class: `card ${cls}`.trim() }, head, children);
}

function chip(text, cls = "") {
  return text ? h("span", { class: `chip ${cls}`.trim() }, text) : null;
}

function kv(label, value) {
  if (value == null || value === "") return null;
  return [h("dt", null, label), h("dd", null, value)];
}

function shareBar(share) {
  const w = ok(share) ? Math.max(0, Math.min(1, num(share))) * 100 : 0;
  return h("span", { class: "share", "aria-hidden": "true" }, h("span", { style: `width:${w.toFixed(1)}%` }));
}

function niceStep(raw) {
  const pow = 10 ** Math.floor(Math.log10(raw || 1));
  const f = raw / pow;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * pow;
}

/** Player rate vs baseline rate, each as a horizontal bar with its 95% interval. */
function compareBars({ rate, ci, base, baseCi, youLabel = "You", baseLabel = "Similar-rated opponents", mini = false }) {
  const values = [rate, base, ...(ci || []), ...(baseCi || [])].map(num).filter(Number.isFinite);
  const max = niceStep(Math.max(0.01, ...values) * 1.08);
  const pos = (v) => (Math.max(0, Math.min(num(v), max)) / max) * 100;
  const lane = (label, value, interval, cls) => {
    const hasCi = Array.isArray(interval) && ok(interval[0]) && ok(interval[1]);
    const title = `${label}: ${pct(value, 1)}${hasCi ? ` (95% interval ${pctRange(interval, 1)})` : ""}`;
    return h("div", { class: `cb-lane ${cls}`, title },
      h("span", { class: "cb-label" }, label),
      h("span", { class: "cb-track" },
        h("span", { class: "cb-bar", style: `width:${Math.max(0.8, pos(value)).toFixed(2)}%` }),
        hasCi && h("span", {
          class: "cb-ci",
          style: `left:${pos(interval[0]).toFixed(2)}%;width:${Math.max(0.5, pos(interval[1]) - pos(interval[0])).toFixed(2)}%`,
        })),
      h("span", { class: "cb-value num" }, pct(value, 1)));
  };
  return h("div", { class: `compare${mini ? " mini" : ""}`, role: "img", "aria-label": `${youLabel} ${pct(rate, 1)} versus ${baseLabel} ${pct(base, 1)}` },
    lane(youLabel, rate, ci, "you"),
    lane(baseLabel, base, baseCi, "base"));
}

function compareLegend() {
  return h("div", { class: "legend" },
    h("span", { class: "item" }, h("span", { class: "share" }, h("span", { style: "width:100%" })), "You"),
    h("span", { class: "item" }, h("span", { class: "share" }, h("span", { style: "width:100%;background:var(--baseline)" })), "Similar-rated opponents (same games)"),
    h("span", { class: "item" }, "Bar: estimated rate · whiskers: 95% interval"));
}

// ================================================================== chess helpers
function parseFenBoard(fen) {
  const board = {};
  String(fen || "").split(" ")[0].split("/").forEach((row, i) => {
    let file = 0;
    for (const ch of row) {
      if (/\d/.test(ch)) file += Number(ch);
      else {
        board[`${"abcdefgh"[file]}${8 - i}`] = ch;
        file += 1;
      }
    }
  });
  return board;
}

/** Convert SAN (e.g. "Nxe5+", "O-O", "e8=Q") to UCI using the list of legal UCI moves. */
function sanToUci(san, fen, legal) {
  if (!san || !Array.isArray(legal)) return null;
  const clean = String(san).trim().replace(/[+#!?]+$/g, "").replace(/0/g, "O");
  const white = String(fen).split(" ")[1] !== "b";
  if (/^O-O(-O)?$/.test(clean)) {
    const rank = white ? "1" : "8";
    const uci = `e${rank}${clean.length > 3 ? "c" : "g"}${rank}`;
    return legal.includes(uci) ? uci : null;
  }
  const m = /^([NBRQK])?([a-h])?([1-8])?x?([a-h][1-8])(?:=?([NBRQnbrq]))?$/.exec(clean);
  if (!m) return null;
  const [, piece = "P", fromFile, fromRank, to, promo] = m;
  const board = parseFenBoard(fen);
  const matches = legal.filter((u) => {
    const p = board[u.slice(0, 2)];
    return u.slice(2, 4) === to && p && p.toUpperCase() === piece &&
      (!fromFile || u[0] === fromFile) && (!fromRank || u[1] === fromRank) &&
      u.slice(4) === (promo ? promo.toLowerCase() : "");
  });
  return matches.length === 1 ? matches[0] : null;
}

/** Accept UCI ("e2e4", "e7e8q") or SAN ("Nf3") typed by the user. */
function parseMoveInput(text, fen, legal) {
  const t = String(text || "").trim();
  if (!t) return null;
  const lower = t.toLowerCase();
  if (/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(lower)) {
    if (legal.includes(lower)) return lower;
    if (lower.length === 4 && legal.includes(`${lower}q`)) return `${lower}q`;
    return null;
  }
  return sanToUci(t, fen, legal);
}

function boardUrl(fen, { orientation = "white", arrow, last, sel, targets } = {}) {
  const params = new URLSearchParams({ fen, orientation });
  if (arrow) params.set("arrow", arrow);
  if (last) params.set("last", last);
  if (sel) params.set("sel", sel);
  if (targets && targets.length) params.set("targets", targets.join(","));
  return `/api/board.svg?${params}`;
}

const sideToMove = (fen) => (String(fen).split(" ")[1] === "b" ? "black" : "white");

/**
 * Read square geometry from a chess.svg board: the viewBox plus every <rect class="square … e4">.
 * Falls back to chess.svg's standard layout (15/390 margin) if the rects are missing.
 */
function parseBoardGeometry(svgText, orientation) {
  const doc = new DOMParser().parseFromString(svgText, "image/svg+xml");
  const root = doc.documentElement;
  const vb = (root.getAttribute("viewBox") || "").trim().split(/[\s,]+/).map(Number);
  const view = vb.length === 4 && vb.every(Number.isFinite)
    ? { x: vb[0], y: vb[1], w: vb[2], h: vb[3] }
    : { x: 0, y: 0, w: num(root.getAttribute("width")) || 390, h: num(root.getAttribute("height")) || 390 };
  let squares = [];
  for (const rect of doc.getElementsByTagName("rect")) {
    const classes = (rect.getAttribute("class") || "").split(/\s+/);
    const name = classes.find((c) => /^[a-h][1-8]$/.test(c));
    if (!classes.includes("square") || !name) continue;
    squares.push({ name, x: num(rect.getAttribute("x")), y: num(rect.getAttribute("y")), w: num(rect.getAttribute("width")), h: num(rect.getAttribute("height")) });
  }
  if (squares.length !== 64 || !squares.every((s) => [s.x, s.y, s.w, s.h].every(Number.isFinite))) {
    const margin = (view.w * 15) / 390;
    const size = (view.w - 2 * margin) / 8;
    squares = [];
    for (let file = 0; file < 8; file++) {
      for (let rank = 0; rank < 8; rank++) {
        const col = orientation === "black" ? 7 - file : file;
        const row = orientation === "black" ? rank : 7 - rank;
        squares.push({ name: `${"abcdefgh"[file]}${rank + 1}`, x: view.x + margin + col * size, y: view.y + margin + row * size, w: size, h: size });
      }
    }
  }
  return { view, squares };
}

function squareAt(geom, img, clientX, clientY) {
  const r = img.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  const x = geom.view.x + ((clientX - r.left) / r.width) * geom.view.w;
  const y = geom.view.y + ((clientY - r.top) / r.height) * geom.view.h;
  const hit = geom.squares.find((s) => x >= s.x && x < s.x + s.w && y >= s.y && y < s.y + s.h);
  return hit ? hit.name : null;
}

// ================================================================== charts
function niceTicks(lo, hi, count = 4) {
  const step = niceStep((hi - lo) / count || 1);
  const start = Math.floor(lo / step) * step;
  const end = Math.ceil(hi / step) * step;
  const ticks = [];
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Math.round(v));
  return { start, end, ticks };
}

/**
 * Rating history: mean line with a shaded 95% band, hover crosshair + tooltip.
 * Re-renders at the container's real width so text stays legible on phones.
 */
function ratingChart(trace, { reference, referenceLabel } = {}) {
  const pts = (trace || []).filter((p) => ok(p.mean) && ok(p.lo) && ok(p.hi));
  if (pts.length < 2) return emptyNote("Not enough games for a rating history yet.");
  const times = pts.map((p) => (p.when ? Date.parse(p.when) : NaN));
  const useTime = times.every(Number.isFinite) && times[times.length - 1] > times[0];
  const xs = useTime ? times : pts.map((_, i) => i);
  const wrap = h("div", { class: "chart" });
  const tooltip = h("div", { class: "tooltip", hidden: true });
  let lastWidth = 0;

  const draw = (width) => {
    const W = Math.max(280, Math.round(width));
    const H = W < 500 ? 200 : 240;
    const m = { l: 44, r: 14, t: 12, b: 26 };
    const lows = pts.map((p) => p.lo).concat(ok(reference) ? [num(reference)] : []);
    const highs = pts.map((p) => p.hi).concat(ok(reference) ? [num(reference)] : []);
    const { start: y0, end: y1, ticks } = niceTicks(Math.min(...lows), Math.max(...highs), 4);
    const x0 = xs[0], x1 = xs[xs.length - 1];
    const X = (v) => m.l + ((v - x0) / (x1 - x0 || 1)) * (W - m.l - m.r);
    const Y = (v) => H - m.b - ((v - y0) / (y1 - y0 || 1)) * (H - m.t - m.b);
    const upper = pts.map((p, i) => `${X(xs[i]).toFixed(1)},${Y(p.hi).toFixed(1)}`);
    const lower = pts.map((p, i) => `${X(xs[i]).toFixed(1)},${Y(p.lo).toFixed(1)}`).reverse();
    const line = pts.map((p, i) => `${X(xs[i]).toFixed(1)},${Y(p.mean).toFixed(1)}`).join(" ");
    const xLabel = (i) => (useTime ? fmtDate(pts[i].when) : `game ${pts[i].n_games ?? i + 1}`);
    const labelIdx = [...new Set([0, Math.floor((pts.length - 1) / 2), pts.length - 1])];
    const cross = svg("line", { class: "cross hover-hidden", y1: m.t, y2: H - m.b });
    const dot = svg("circle", { class: "dot hover-hidden", r: 4.5 });

    const chart = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": `Rating estimate over time, from ${fmt(pts[0].mean)} to ${fmt(pts[pts.length - 1].mean)}` },
      ticks.map((t) => [
        svg("line", { class: "grid-line", x1: m.l, x2: W - m.r, y1: Y(t), y2: Y(t) }),
        svg("text", { class: "axis-text", x: m.l - 8, y: Y(t) + 4, "text-anchor": "end" }, fmt(t)),
      ]),
      svg("polygon", { class: "band", points: upper.concat(lower).join(" ") }),
      ok(reference) && [
        svg("line", { class: "ref-line", x1: m.l, x2: W - m.r, y1: Y(reference), y2: Y(reference) }),
        svg("text", { class: "ref-text", x: W - m.r, y: Y(reference) - 5, "text-anchor": "end" }, referenceLabel || fmt(reference)),
      ],
      svg("polyline", { class: "line", points: line }),
      labelIdx.map((i, k) => svg("text", {
        class: "axis-text", x: X(xs[i]), y: H - 6,
        "text-anchor": k === 0 ? "start" : i === pts.length - 1 ? "end" : "middle",
      }, xLabel(i))),
      cross, dot);

    const overlay = svg("rect", { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: "transparent" });
    chart.append(overlay);
    const hide = () => {
      cross.classList.add("hover-hidden");
      dot.classList.add("hover-hidden");
      tooltip.hidden = true;
    };
    overlay.addEventListener("pointerleave", hide);
    overlay.addEventListener("pointermove", (ev) => {
      const box = chart.getBoundingClientRect();
      const px = ((ev.clientX - box.left) / box.width) * W;
      let best = 0;
      for (let i = 1; i < pts.length; i++) if (Math.abs(X(xs[i]) - px) < Math.abs(X(xs[best]) - px)) best = i;
      const p = pts[best];
      const cx = X(xs[best]), cy = Y(p.mean);
      cross.setAttribute("x1", cx);
      cross.setAttribute("x2", cx);
      dot.setAttribute("cx", cx);
      dot.setAttribute("cy", cy);
      cross.classList.remove("hover-hidden");
      dot.classList.remove("hover-hidden");
      tooltip.replaceChildren(
        h("span", { class: "t-sub" }, p.when ? fmtDate(p.when) : `Game ${best + 1}`),
        h("span", null, "Estimate ", h("b", null, fmt(p.mean))),
        h("span", null, `95% interval ${range(p.lo, p.hi)}`),
        ok(p.n_games) && h("span", { class: "t-sub" }, `after ${plural(p.n_games, "game")}`));
      tooltip.hidden = false;
      const scale = box.width / W;
      const left = cx * scale + 12;
      tooltip.style.left = `${Math.min(left, box.width - 170)}px`;
      tooltip.style.top = `${Math.max(0, cy * scale - 40)}px`;
    });
    wrap.replaceChildren(chart, tooltip);
  };

  const observer = new ResizeObserver((entries) => {
    if (!wrap.isConnected) {
      observer.disconnect();
      return;
    }
    const width = entries[0].contentRect.width;
    if (width > 0 && Math.abs(width - lastWidth) > 4) {
      lastWidth = width;
      draw(width);
    }
  });
  observer.observe(wrap);

  const table = h("details", null,
    h("summary", null, "Show data table"),
    h("div", { class: "data-table-scroll" },
      h("table", { class: "data" },
        h("thead", null, h("tr", null, h("th", null, "Date"), h("th", { class: "r" }, "Estimate"), h("th", { class: "r" }, "95% interval"), h("th", { class: "r" }, "Games"))),
        h("tbody", null, pts.slice().reverse().map((p) => h("tr", null,
          h("td", null, p.when ? fmtDate(p.when) : "–"),
          h("td", { class: "r num" }, fmt(p.mean)),
          h("td", { class: "r num" }, range(p.lo, p.hi)),
          h("td", { class: "r num" }, fmt(p.n_games))))))));
  return h("div", { class: "stack" }, wrap, table);
}

// ================================================================== Overview
function renderOverview() {
  const summary = shared(() => api("/api/summary").then((s) => { setPlayerName(s); return s; }));
  const slow = "Building your profile… the first load can take up to a minute.";
  const section = (title, render, opts = {}) =>
    card({ title, ...opts }, asyncBlock(summary, render, slow));

  return h("div", { class: "view" },
    pageHead("Overview", "Where your play stands, compared with similar-rated opponents in the same games."),
    section("Player profile", profileBlock),
    section("Style", styleBlock, { sub: "how often you choose an option when it is available" }),
    h("div", { class: "grid-2" },
      section("Strengths", (s) => strengthList(s.summary && s.summary.strengths, "strength")),
      section("Weaknesses", (s) => strengthList(s.summary && s.summary.weaknesses, "weakness"),
        { action: h("a", { href: "#weaknesses", class: "small" }, "All patterns →") })),
    h("div", { class: "grid-2" },
      card({ title: "Openings", action: h("a", { href: "#openings", class: "small" }, "Full repertoire →") },
        asyncBlock(() => api("/api/openings"), openingsMini)),
      section("Recent development", developmentBlock)),
    h("div", { class: "grid-2" },
      section("Mirror accuracy", mirrorAccuracyBlock, { action: h("a", { href: "#mirror", class: "small" }, "Details →") }),
      section("Training", trainingSummaryBlock)),
    tierLegend());
}

function profileBlock(s) {
  const rating = s.rating || {};
  const pool = rating.primary_pool;
  const combined = pool && rating.combined ? rating.combined[pool] : null;
  const results = pool && rating.results ? rating.results[pool] : null;
  const current = results && results.current;
  const elo = combined || current;
  const strength = s.strength || {};
  const acc = Array.isArray(strength.accuracy) ? strength.accuracy : [];
  const overview = s.overview || {};

  const tile = (label, value, ...sub) =>
    h("div", { class: "tile" }, h("span", { class: "tile-label" }, label), h("span", { class: "tile-value num" }, value), sub.filter(Boolean).map((t) => h("span", { class: "tile-sub" }, t)));

  const facts = [];
  if (ok(strength.blunders_per_100 && strength.blunders_per_100[0])) {
    facts.push(`Per 100 moves: ${fmt(strength.blunders_per_100[0], 1)} blunders, ${fmt(strength.mistakes_per_100[0], 1)} mistakes, ${fmt(strength.inaccuracies_per_100[0], 1)} inaccuracies`);
  }
  const res = s.results || {};
  if (res.conversion && ok(res.conversion.n)) facts.push(`won ${res.conversion.k} of ${res.conversion.n} games after reaching a winning position`);
  if (res.saves && ok(res.saves.n)) facts.push(`saved ${res.saves.k} of ${res.saves.n} after falling to a losing one`);

  return [
    h("div", { class: "tiles" },
      tile("Estimated Elo", elo ? fmt(elo.mean) : "–",
        elo ? `95% interval ${range(elo.lo, elo.hi)}` : "not enough rated games yet",
        pool ? `${poolName(pool)}${combined ? " · results + move quality" : " · from results"}` : null,
        current ? `${plural(current.n_games, "rated game")}` : null),
      tile("Accuracy", ok(acc[0]) ? fmt(acc[0], 1) : "–", ok(acc[1]) ? `95% CI ${range(acc[1], acc[2], 1)}` : null,
        ok(strength.moves) ? `${fmt(strength.moves)} non-forced moves` : null),
      tile("Games analysed", fmt(overview.games), "with engine analysis"),
      tile("Positions analysed", fmt(overview.positions), ok(overview.population_positions) ? `+ ${fmt(overview.population_positions)} opponent moves as baseline` : null)),
    facts.length ? h("p", { class: "facts-line" }, `${capitalize(facts.join(" · "))} `, tierBadge(FACT)) : null,
  ];
}

function styleBlock(s) {
  const traits = Object.entries(s.style || {}).filter(([, rows]) => Array.isArray(rows) && rows.length);
  if (!traits.length) return emptyNote("No style data yet — analyse more games.");
  return [
    h("div", { class: "traits" }, traits.map(([trait, rows]) =>
      h("div", { class: "trait" }, h("h3", null, trait), rows.map(styleRow)))),
    compareLegend(),
  ];
}

function styleRow(r) {
  const weak = isInsufficient(r.tier);
  const direction = weak ? "no clear difference" : r.direction === "higher" ? "more often than opponents" : "less often than opponents";
  return h("div", { class: `trait-row${weak ? " is-muted" : ""}` },
    h("div", { class: "what" }, capitalize(r.label),
      h("span", { class: "dir" }, `${pct(r.rate)} vs ${pct(r.baseline_rate)} · ${direction} · n=${fmt(r.n)}`)),
    compareBars({ rate: r.rate, ci: r.ci, base: r.baseline_rate, baseCi: r.baseline_ci, mini: true }),
    h("div", { class: "tier-cell" }, tierBadge(r.tier)));
}

function strengthList(items, kind) {
  const list = (items || []).filter((x) => !isInsufficient(x.tier));
  if (!list.length) return emptyNote(`No ${kind} with sufficient evidence yet.`);
  const row = (x) => h("li", null,
    h("span", { class: "label" }, capitalize(x.label)),
    h("span", { class: "meta" }, h("span", { class: "num", title: "Probability the difference is real" }, `${pct(x.confidence)} conf.`),
      h("span", { class: "num" }, `n=${fmt(x.n)}`), tierBadge(x.tier)));
  const first = list.slice(0, 6), rest = list.slice(6);
  return [
    h("ul", { class: "item-list" }, first.map(row)),
    rest.length ? h("details", null, h("summary", null, `Show ${rest.length} more`), h("ul", { class: "item-list" }, rest.map(row))) : null,
  ];
}

function openingsMini(data) {
  const col = (color) => {
    const d = data && data[color];
    const rep = d && Array.isArray(d.repertoire) ? d.repertoire.slice(0, 3) : [];
    return h("div", { class: "stack" },
      h("h3", { class: "small" }, `As ${capitalize(color)}`, d ? h("span", { class: "muted-text" }, ` · ${plural(d.games, "game")}`) : null),
      rep.length
        ? h("ul", { class: "item-list" }, rep.map((r) => h("li", null,
          h("span", { class: "label" }, r.opening),
          h("span", { class: "meta" }, shareBar(r.share), h("span", { class: "num" }, `${pct(r.share)} · ${r.games}`)))))
        : emptyNote("No games yet."));
  };
  return h("div", { class: "stack" }, col("white"), col("black"));
}

function developmentBlock(s) {
  const dev = s.development || {};
  const metrics = Object.entries(dev.metrics || {});
  if (!metrics.length) return emptyNote(dev.note || "Not enough games to compare recent and earlier play.");
  return [
    h("p", { class: "note" }, `Your most recent ${plural(dev.recent_games, "game")} compared with the ${fmt(dev.earlier_games)} before.`),
    h("ul", { class: "item-list" }, metrics.map(([name, m]) => {
      const weak = isInsufficient(m.tier);
      const trend = weak ? "no clear change" : m.trend;
      return h("li", { class: weak ? "is-muted" : "" },
        h("span", { class: "label" }, capitalize(name),
          h("span", { class: "tile-sub", style: "display:block" }, `${pct(m.recent, 1)} recently vs ${pct(m.earlier, 1)} before`)),
        h("span", { class: "meta" },
          weak ? h("span", null, trend) : chip(capitalize(trend), m.trend === "improving" ? "chip-good" : "chip-bad"),
          tierBadge(m.tier)));
    })),
  ];
}

// ---- Mirror helpers shared by Overview and the Mirror tab
const MODEL_INFO = {
  "random-legal": "Picks any legal move at random",
  "stockfish-best": "Always plays the engine's best move",
  "generic Elo-matched strength only": "Move quality only, fitted on your opponents",
  "population (Elo-matched opponents)": "Full model fitted on your similar-rated opponents",
  "my strength only (no style)": "Move quality only, fitted on your games",
  "personal logit (no book)": "Your personal model, without your opening book",
};
const mirrorKey = (metrics) => Object.keys(metrics || {}).find((k) => k.startsWith("mirror"));

function significance(sig, against) {
  if (!sig || !ok(sig.mean_diff)) return null;
  const [lo, hi] = sig.ci95 || [];
  const d = num(sig.mean_diff);
  const p = num(sig.p_a_better);
  let verdict, tier;
  if (ok(hi) && hi < 0) [verdict, tier] = [`Mirror predicts your moves better than ${against}.`, INFERENCE];
  else if (ok(lo) && lo > 0) [verdict, tier] = [`Mirror predicts your moves worse than ${against}.`, INFERENCE];
  else [verdict, tier] = [`No clear difference from ${against} yet.`, p >= 0.75 || p <= 0.25 ? HYPOTHESIS : INSUFFICIENT];
  const detail = `Log loss ${d <= 0 ? "lower" : "higher"} by ${Math.abs(d).toFixed(3)} (95% interval of the difference ${signed(lo, 3)} to ${signed(hi, 3)}); chance Mirror is really better: ${probText(p)}.`;
  return h("div", { class: "stack", style: "gap:4px" }, h("div", { class: "row" }, h("strong", null, verdict), tierBadge(tier)), h("p", { class: "note" }, detail));
}

function metricsTable(metrics, names, { describe = false } = {}) {
  const mk = mirrorKey(metrics);
  const rows = names.filter((n) => metrics[n] && !n.startsWith("_"));
  return h("div", { class: "table-wrap" }, h("table", { class: "data" },
    h("thead", null, h("tr", null,
      h("th", null, "Model"), h("th", { class: "r" }, "Top-1"), h("th", { class: "r" }, "Top-3"), h("th", { class: "r" }, "Log loss"),
      describe && h("th", { class: "r hide-sm" }, "Perplexity"), describe && h("th", { class: "r hide-sm" }, "Calib. error"),
      describe && h("th", { class: "r hide-sm" }, "Moves"))),
    h("tbody", null, rows.map((name) => {
      const m = metrics[name];
      const isMirror = name === mk;
      return h("tr", { class: isMirror ? "highlight" : "" },
        h("td", null, isMirror ? h("strong", null, name) : name,
          describe && h("span", { class: "sub" }, isMirror ? "Personal model + your opening book (what the engine plays)" : MODEL_INFO[name] || "")),
        h("td", { class: "r num" }, pct(m.top1, 1)),
        h("td", { class: "r num" }, pct(m.top3, 1)),
        h("td", { class: "r num" }, fmt(m.log_loss, 3)),
        describe && h("td", { class: "r num hide-sm" }, fmt(m.perplexity, 1)),
        describe && h("td", { class: "r num hide-sm" }, fmt(m.ece, 3)),
        describe && h("td", { class: "r num hide-sm" }, fmt(m.n)));
    }))));
}

function mirrorAccuracyBlock(s) {
  const model = s.mirror;
  if (!model) return emptyNote("No active Mirror model yet. Train one with `mirror train` (one may be training right now).");
  const metrics = model.metrics || {};
  const mk = mirrorKey(metrics);
  if (!mk) return emptyNote(`Model ${model.version} has no held-out evaluation.`);
  const sig = metrics._significance || {};
  return [
    h("p", { class: "note" }, h("strong", null, model.version), ` · evaluated on ${fmt(metrics[mk].n)} moves from held-out games it never saw. Higher top-k and lower log loss are better.`),
    metricsTable(metrics, [mk, "population (Elo-matched opponents)", "my strength only (no style)"]),
    significance(sig.mirror_vs_population, "the population model"),
    significance(sig.mirror_vs_my_strength, "a strength-only model"),
  ];
}

function trainingSummaryBlock(s) {
  const t = s.training || {};
  const cats = Object.entries(t.by_category || {});
  const due = cats.reduce((a, [, v]) => a + (num(v.due) || 0), 0);
  const button = h("a", { class: "btn btn-primary", href: "#training" }, due ? `Train ${plural(due, "position")}` : "Open trainer");
  if (!cats.length) return [emptyNote("No training positions yet. Build them from your mistakes in the Training tab."), h("div", null, button)];
  return [
    h("p", null, h("strong", { class: "num" }, fmt(due)), ` ${due === 1 ? "position is" : "positions are"} due for review.`),
    h("ul", { class: "item-list" }, cats.map(([name, v]) => h("li", null,
      h("span", { class: "label" }, name),
      h("span", { class: "meta num" }, `${fmt(v.due || 0)} due / ${fmt(v.total)}`)))),
    ok(t.reviews) && t.reviews > 0 ? h("p", { class: "note" }, `${plural(t.reviews, "review")} so far, ${fmt(t.correct)} correct; ${plural(t.failed_then_solved || 0, "position")} failed at first and solved later.`) : null,
    h("div", null, button),
  ];
}

// ================================================================== Weaknesses
function renderWeaknesses() {
  return h("div", { class: "view" },
    pageHead("Weaknesses", "Every pattern tracked across your games. Each compares how often you fail when the situation arises with how often your similar-rated opponents fail in the same games."),
    asyncBlock(() => api("/api/weaknesses"), (list) => {
      if (!Array.isArray(list) || !list.length) return emptyNote("No patterns yet — analyse some games first.");
      const weak = list.filter((w) => w.is_weakness);
      const other = list.filter((w) => !w.is_weakness);
      return [
        h("p", { class: "lead" }, `${weak.length} of ${list.length} tracked patterns currently look like weaknesses.`),
        compareLegend(),
        weak.length ? h("div", { class: "cards" }, weak.map(weaknessCard)) : emptyNote("No weakness with sufficient evidence yet."),
        other.length ? card({}, h("details", null,
          h("summary", null, `Other tracked patterns (${other.length}) — no weakness, or not enough evidence`),
          h("div", { class: "cards" }, other.map(weaknessCard)))) : null,
      ];
    }, "Counting patterns across your games…"));
}

function weaknessCard(w) {
  const o = w.overall || {};
  const recent = w.recent || {};
  const t = w.trend;
  const muted = isInsufficient(o.tier);
  let trend;
  if (!t) trend = h("span", { class: "muted-text" }, "Trend: not enough recent data");
  else if (isInsufficient(t.tier)) trend = h("span", { class: "row" }, h("span", { class: "muted-text" }, `Trend: no clear change (${pct(t.recent_rate)} recently vs ${pct(t.earlier_rate)} before)`), tierBadge(t.tier));
  else trend = h("span", { class: "row" }, "Trend: ", chip(capitalize(t.direction), t.direction === "improving" ? "chip-good" : "chip-bad"),
    h("span", { class: "num" }, `${pct(t.recent_rate)} recently vs ${pct(t.earlier_rate)} before`), tierBadge(t.tier));

  return h("article", { class: `card wcard${muted ? " is-muted" : ""}` },
    h("div", { class: "wcard-head" },
      h("h3", null, capitalize(w.label)),
      h("div", { class: "row" }, chip(w.category), tierBadge(o.tier),
        !muted && ok(o.confidence) ? h("span", { class: "note num" }, `${pct(o.confidence)} confidence`) : null)),
    w.text ? h("p", { class: "text" }, w.text) : null,
    compareBars({
      rate: o.rate, ci: o.ci, base: o.baseline_rate, baseCi: o.baseline_ci,
      youLabel: `You · ${fmt(o.k)}/${fmt(o.n)}`, baseLabel: `Opponents · ${fmt(o.baseline_k)}/${fmt(o.baseline_n)}`,
    }),
    h("div", { class: "foot" },
      h("span", { class: "row" }, `Last ${plural(recent.games, "game")}: ${fmt(recent.failures)} of ${fmt(recent.opportunities)} ${w.unit || ""}`, tierBadge(FACT)),
      trend));
}

// ================================================================== Openings
function renderOpenings() {
  return h("div", { class: "view" },
    pageHead("Openings", "Your repertoire, the moves you choose, and the opening mistakes that keep coming back. The Mirror engine plays these same frequencies."),
    asyncBlock(() => api("/api/openings"), (data) => ["white", "black"].map((c) => openingColor(c, data && data[c])),
      "Reading your opening moves…"));
}

const REPERTOIRE_SHOWN = 8;

/** Button that toggles the "expanded" class on its card, revealing rows marked .extra. */
function expandToggle(moreText, lessText) {
  const btn = h("button", { class: "btn btn-small btn-quiet", type: "button", "aria-expanded": "false" }, moreText);
  btn.addEventListener("click", () => {
    const expanded = btn.closest(".card").classList.toggle("expanded");
    btn.textContent = expanded ? lessText : moreText;
    btn.setAttribute("aria-expanded", String(expanded));
  });
  return h("div", null, btn);
}

function openingColor(color, d) {
  const title = `As ${capitalize(color)}`;
  if (!d || !d.games) return card({ title }, emptyNote("No games with this colour yet."));
  const rep = d.repertoire || [];
  return card({ title, sub: plural(d.games, "game") },
    d.most_comfortable && d.most_comfortable.length
      ? h("div", { class: "row small" }, h("span", { class: "muted-text" }, "Highest accuracy (5+ games):"), d.most_comfortable.map((n) => chip(n, "chip-accent")))
      : null,
    rep.length ? h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", null, h("tr", null,
        h("th", null, "Opening"), h("th", { class: "r" }, "Games"), h("th", null, "Share"), h("th", { class: "r" }, "Score (95% CI)"),
        h("th", { class: "r hide-sm" }, "Opening accuracy"), h("th", { class: "r hide-sm" }, "Mistakes / game"))),
      h("tbody", null, rep.map((r, i) => h("tr", { class: i >= REPERTOIRE_SHOWN ? "extra" : "" },
        h("td", null, r.opening),
        h("td", { class: "r num" }, fmt(r.games)),
        h("td", { class: "num" }, shareBar(r.share), " ", pct(r.share)),
        h("td", { class: "r num" }, pct(r.score), " ", h("span", { class: "muted-text small" }, pctRange(r.score_ci))),
        h("td", { class: "r num hide-sm" }, fmt(r.accuracy, 1)),
        h("td", { class: "r num hide-sm" }, fmt(r.mistakes_per_game, 2)))))))
      : emptyNote("No repertoire data."),
    rep.length > REPERTOIRE_SHOWN ? expandToggle(`Show all ${rep.length} openings`, "Show fewer") : null,
    h("h3", null, "Your choices"),
    d.choices && d.choices.length
      ? h("div", { class: "choices" }, d.choices.map((c) => h("div", { class: "choice" },
        h("span", { class: "after" }, c.after === "(start)" ? "From the starting position" : `After ${numberLine(c.after)}`, ` · n=${fmt(c.n)}`),
        h("div", { class: "moves" }, (c.moves || []).map((m) => h("span", { class: "move-share", title: `${m.count} of ${c.n} games` },
          h("b", null, m.san), shareBar(m.share), h("span", { class: "num" }, pct(m.share))))))))
      : emptyNote("Not enough repeated positions yet."),
    h("h3", null, "Recurring opening mistakes"),
    d.recurring_opening_mistakes && d.recurring_opening_mistakes.length
      ? h("ul", { class: "item-list" }, d.recurring_opening_mistakes.map((e) => {
        const plies = String(e.line || "").split(/\s+/).filter(Boolean);
        const moveNo = Math.floor(plies.length / 2) + 1;
        const played = `${moveNo}${plies.length % 2 === 0 ? "." : "..."}${e.played}`;
        return h("li", null,
          h("span", { class: "label" }, plies.length ? `After ${numberLine(plies)} you played ` : "From the start you played ", h("strong", null, played),
            e.better ? [" — better was ", h("strong", null, e.better)] : null),
          h("span", { class: "meta" }, h("span", { class: "num" }, `${e.times}×`), tierBadge(FACT)));
      }))
      : emptyNote("No opening mistake repeated in the same position (yet)."));
}

// ================================================================== Games & Coach
function renderGames(gameId) {
  return gameId ? renderCoachReview(gameId) : renderGameList();
}

function playerOutcome(g) {
  const r = g.result;
  if (r === "1/2-1/2") return ["Draw", ""];
  if (r !== "1-0" && r !== "0-1") return [r || "?", ""];
  const won = (r === "1-0") === (g.player_color === "white");
  return won ? ["Win", "chip-good"] : ["Loss", "chip-bad"];
}

function renderGameList() {
  return h("div", { class: "view" },
    pageHead("Games & Coach", "Your most recent games. Open one for a coach review of the moves that mattered most."),
    card({}, asyncBlock(() => api("/api/games"), (games) => {
      if (!Array.isArray(games) || !games.length) return emptyNote("No games imported yet.");
      return [
        h("div", { class: "table-wrap" }, h("table", { class: "data" },
          h("thead", null, h("tr", null,
            h("th", null, "Date"), h("th", null, "Opponent"), h("th", { class: "hide-sm" }, "Colour"), h("th", null, "Result"),
            h("th", { class: "r" }, "Accuracy"), h("th", { class: "r hide-sm" }, "Blunders / mistakes"), h("th", { class: "hide-sm" }, ""))),
          h("tbody", null, games.map(gameRow)))),
        h("p", { class: "note" }, `Showing the ${games.length} most recent games. “Held out” games were never used to train the Mirror; they measure how well it predicts you.`),
      ];
    }, "Loading games…")));
}

function gameRow(g) {
  const opponent = g.player_color === "white" ? g.black : g.white;
  const [outcome, cls] = playerOutcome(g);
  const go = () => { location.hash = `#games/${g.id}`; };
  return h("tr", { class: "clickable", onclick: go },
    h("td", { class: "num" }, fmtDate(g.date)),
    h("td", null, h("a", { href: `#games/${g.id}`, onclick: (e) => e.stopPropagation() }, opponent || "?"),
      h("span", { class: "sub" }, [g.tc_class, g.opening].filter(Boolean).join(" · "))),
    h("td", { class: "hide-sm" }, capitalize(g.player_color)),
    h("td", null, chip(outcome, cls), h("span", { class: "sub num" }, g.result)),
    h("td", { class: "r num" }, g.analyzed ? fmt(g.accuracy, 1) : h("span", { class: "muted-text" }, "not analysed")),
    h("td", { class: "r num hide-sm" }, g.analyzed ? `${fmt(g.blunders)} / ${fmt(g.mistakes)}` : "–"),
    h("td", { class: "hide-sm" }, g.split === "test" ? h("span", { class: "chip chip-accent", title: "Held out from Mirror training; used to test it" }, "held out") : null));
}

function renderCoachReview(gameId) {
  return h("div", { class: "view" },
    h("a", { href: "#games", class: "back-link" }, "← All games"),
    asyncBlock(() => api(`/api/coach/${encodeURIComponent(gameId)}`), coachReview, "Preparing the review…"));
}

function coachReview(rv) {
  const color = rv.player_color || "white";
  const c = rv.counts || {};
  const [outcome, cls] = playerOutcome(rv);
  const issues = rv.issues || [];
  if (rv.analyzed === false) {
    return [
      pageHead(`${rv.white || "?"} – ${rv.black || "?"}`, null),
      h("div", { class: "row" }, chip(outcome, cls), chip(rv.result), chip(fmtDate(rv.date)), chip(`You played ${color}`)),
      card({ title: "Not analysed yet" },
        h("p", null, "Stockfish hasn't analysed this game yet, so there is nothing to review. Games you play in the app are analysed automatically after they end; imported games are analysed by ",
          h("a", { href: "#setup" }, "Update everything"), "."),
        h("p", { class: "note" }, `${plural(rv.player_moves, "move")} of yours waiting for analysis.`)),
    ];
  }
  return [
    pageHead(`${rv.white || "?"} – ${rv.black || "?"}`, null),
    h("div", { class: "row" }, chip(outcome, cls), chip(rv.result), chip(fmtDate(rv.date)), chip(`You played ${color}`), rv.opening ? chip(rv.opening) : null),
    card({ title: "Summary" },
      h("p", null, `Accuracy ${fmt(rv.accuracy, 1)} · ${plural(c.blunders, "blunder")}, ${plural(c.mistakes, "mistake")}, ${plural(c.inaccuracies, "inaccuracy", "inaccuracies")}. `,
        issues.length ? `Reviewing the ${issues.length === 1 ? "most costly moment" : `${issues.length} most costly moments`}, largest first.` : "No significant mistakes — nothing lost 10%+ win chance in an undecided position. Well played."),
      rv.minor_issues_skipped ? h("p", { class: "note" }, `${plural(rv.minor_issues_skipped, "smaller issue")} not reviewed, to keep the focus on what matters.`) : null),
    issues.map((it, i) => issueCard(it, i, color)),
  ];
}

function issueCard(it, index, color) {
  const img = h("img", { class: "board-img", alt: `Position before ${it.move_label}${it.played}`, width: 480, height: 480, loading: "lazy" });
  const caption = h("p", { class: "note" }, "Position before your move.");
  api(`/api/legal?fen=${encodeURIComponent(it.fen)}`)
    .then((legal) => sanToUci(it.best, it.fen, legal))
    .catch(() => null)
    .then((best) => {
      img.src = boardUrl(it.fen, { orientation: color, arrow: best });
      if (best) caption.textContent = `Position before your move. Arrow: best move ${it.best}.`;
    });

  const wpDelta = ok(it.wp_after) && ok(it.wp_before) ? (it.wp_after - it.wp_before) * 100 : NaN;
  const evalText = (cp) => (ok(cp) ? signed(cp / 100, 1) : "?");
  const timeText = ok(it.time_spent)
    ? `${fmt(it.time_spent)}s spent${ok(it.clock) ? ` · ${fmtClock(it.clock)} left on the clock` : ""}`
    : null;

  return h("article", { class: "card issue" },
    h("div", { class: "issue-board" }, img, caption),
    h("div", { class: "issue-body" },
      h("div", { class: "issue-title" },
        h("span", { class: "moveno" }, `${index + 1}. ${it.move_label}${it.played}`),
        chip(capitalize(it.category), "chip-bad"),
        it.motif ? chip(it.motif.replace(/_/g, " ")) : null),
      h("dl", { class: "kv" },
        kv("Your move / best", h("span", null, h("strong", null, it.played), " — best was ", h("strong", null, it.best))),
        kv("Evaluation (your side)", h("span", { class: "num" }, `${evalText(it.eval_before)} → ${evalText(it.eval_after)}`)),
        kv("Win chance", h("span", { class: "num" }, `${pct(it.wp_before)} → ${pct(it.wp_after)} `,
          ok(wpDelta) ? h("span", { class: "delta-bad" }, `(${signed(wpDelta, 0)} points)`) : null)),
        kv("Why", (it.why || []).join(" ")),
        kv("What you probably overlooked", it.overlooked),
        kv("Line after your move", it.refutation_line ? h("span", { class: "moves-line" }, it.refutation_line) : null),
        kv("Better", it.best_line ? h("span", { class: "moves-line" }, it.best_line) : null),
        kv("Concept", it.concept ? h("span", null, h("strong", null, it.concept.name), it.concept.idea ? ` — ${it.concept.idea}` : "") : null),
        kv("Recognise it next time", it.recognize),
        kv("Time", timeText))));
}

// ================================================================== Training
function renderTraining() {
  const state = {
    item: null, legal: [], sel: null, targets: [], lastMove: null, arrow: null, geom: null,
    started: 0, result: null, busy: false, category: "", drawToken: 0, blobUrl: null, next: null,
  };
  const img = h("img", { class: "board-img", alt: "Training position — click a piece, then its destination", width: 480, height: 480, draggable: "false" });
  const boardWrap = h("div", { class: "board-wrap" }, img);
  const boardNote = h("p", { class: "note" });
  const boardCol = h("div", { class: "stack" }, boardWrap, boardNote);
  const panel = h("div", { class: "card stack" });
  const layout = h("div", { class: "trainer" }, boardCol, panel);
  const statsCard = h("section", { class: "card" }, loadingEl());
  const categorySelect = h("select", { "aria-label": "Category", onchange: () => { state.category = categorySelect.value; loadNext(); } },
    h("option", { value: "" }, "All categories"));
  const buildStatus = h("span", { class: "note", role: "status" });

  img.addEventListener("click", onBoardClick);

  loadNext();
  return h("div", { class: "view" },
    pageHead("Training", "Positions from your own games where you went wrong. Correct answers come back later; misses come back tomorrow."),
    h("div", { class: "row" }, h("label", { class: "row small" }, "Practise:", categorySelect), buildButton(), buildStatus),
    layout,
    statsCard);

  // -------------------------------------------------------------- flow
  async function loadNext(payload) {
    Object.assign(state, { item: null, legal: [], sel: null, targets: [], lastMove: null, arrow: null, result: null, next: null });
    boardWrap.classList.remove("locked");
    panel.replaceChildren(loadingEl("Loading the next position…"));
    try {
      const qs = state.category ? `?category=${encodeURIComponent(state.category)}` : "";
      const data = payload || (await api(`/api/training/next${qs}`));
      fillCategories(data.categories || {});
      renderStats(data.stats || {});
      state.item = data.item;
      if (!state.item) {
        layout.classList.add("single");
        boardCol.hidden = true;
        panel.replaceChildren(...nothingDue(data.stats || {}));
        return;
      }
      layout.classList.remove("single");
      boardCol.hidden = false;
      state.legal = await api(`/api/legal?fen=${encodeURIComponent(state.item.fen)}`);
      await drawBoard();
      state.started = performance.now();
      renderPanel();
    } catch (err) {
      panel.replaceChildren(errorBox(err, () => loadNext()));
    }
  }

  async function drawBoard() {
    const it = state.item;
    if (!it) return;
    const token = ++state.drawToken;
    const res = await fetch(boardUrl(it.fen, { orientation: it.orientation, sel: state.sel, targets: state.targets, last: state.lastMove, arrow: state.arrow }));
    if (!res.ok) throw new Error(`board image failed (${res.status})`);
    const text = await res.text();
    if (token !== state.drawToken) return;
    state.geom = parseBoardGeometry(text, it.orientation);
    const old = state.blobUrl;
    state.blobUrl = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
    img.src = state.blobUrl;
    if (old) setTimeout(() => URL.revokeObjectURL(old), 2000);
    boardNote.textContent = state.result
      ? (state.arrow ? `Arrow: best move ${state.result.best_san}. Highlighted: your answer.` : "Highlighted: your answer.")
      : `${capitalize(it.orientation)} to move — board shown from ${it.orientation}'s side.`;
  }

  function onBoardClick(ev) {
    if (!state.item || state.result || state.busy || !state.geom) return;
    const sq = squareAt(state.geom, img, ev.clientX, ev.clientY);
    if (!sq) return;
    if (state.sel && state.targets.includes(sq)) {
      submit(chooseMove(state.sel, sq));
      return;
    }
    const from = state.legal.filter((u) => u.startsWith(sq));
    if (from.length && sq !== state.sel) {
      state.sel = sq;
      state.targets = [...new Set(from.map((u) => u.slice(2, 4)))];
    } else {
      state.sel = null;
      state.targets = [];
    }
    drawBoard().catch((err) => panel.append(errorBox(err)));
  }

  function chooseMove(from, to) {
    const options = state.legal.filter((u) => u.startsWith(from + to));
    return options.find((u) => u.length === 4) || options.find((u) => u.endsWith("q")) || options[0];
  }

  async function submit(uci) {
    if (!uci || state.busy || state.result) return;
    state.busy = true;
    boardWrap.classList.add("locked");
    const seconds = Math.round((performance.now() - state.started) / 100) / 10;
    try {
      const res = await api("/api/training/answer", { method: "POST", body: { id: state.item.id, uci, seconds } });
      state.result = { ...res, uci, seconds };
      state.sel = null;
      state.targets = [];
      state.lastMove = uci;
      state.arrow = sanToUci(res.best_san, state.item.fen, state.legal);
      renderPanel();
      drawBoard().catch(() => {});
      prefetchNext();
    } catch (err) {
      boardWrap.classList.remove("locked");
      panel.append(errorBox(err));
    } finally {
      state.busy = false;
    }
  }

  async function prefetchNext() {
    try {
      const qs = state.category ? `?category=${encodeURIComponent(state.category)}` : "";
      const data = await api(`/api/training/next${qs}`);
      state.next = data;
      renderStats(data.stats || {});
    } catch {
      state.next = null; // "Next" will simply fetch again
    }
  }

  // -------------------------------------------------------------- panel
  function renderPanel() {
    const it = state.item;
    panel.replaceChildren(
      h("div", { class: "row" }, chip(it.category, "chip-accent"),
        it.reps ? chip(`${plural(it.reps, "correct review")} in a row`) : null,
        it.lapses ? chip(plural(it.lapses, "miss", "misses")) : null),
      h("p", { class: "prompt" }, it.prompt),
      state.result ? resultBlock(state.result) : moveForm());
  }

  function moveForm() {
    const input = h("input", { type: "text", placeholder: "e.g. e2e4 or Nf3", autocomplete: "off", autocapitalize: "off", spellcheck: "false", "aria-label": "Type your move" });
    const error = h("p", { class: "form-error", role: "alert" });
    const form = h("form", {
      class: "inline-form",
      onsubmit: (e) => {
        e.preventDefault();
        const uci = parseMoveInput(input.value, state.item.fen, state.legal);
        if (!uci) {
          error.textContent = "That isn't a legal move here. Use from-to squares like e2e4 (add q/r/b/n to promote), or SAN like Nf3.";
          return;
        }
        submit(uci);
      },
    }, input, h("button", { class: "btn", type: "submit" }, "Submit"));
    return h("div", { class: "stack" },
      h("p", { class: "hint" }, "Click a piece, then the square to move it to. Pawns reaching the last rank become queens. Or type the move:"),
      form, error);
  }

  function resultBlock(r) {
    const next = h("button", { class: "btn btn-primary", type: "button", onclick: () => loadNext(state.next || undefined) }, "Next position");
    setTimeout(() => next.focus({ preventScroll: true }), 0);
    return h("div", { class: `result ${r.correct ? "ok" : "fail"}` },
      h("div", { class: "result-head" }, r.correct ? "Correct" : "Not quite"),
      h("dl", { class: "kv" },
        kv("Your answer", h("span", { class: "mono" }, r.uci)),
        kv("Best move", h("strong", null, r.best_san)),
        kv("Accepted moves", (r.accepted || []).join(", ") || null),
        kv("In the game you played", r.played_in_game),
        kv("Time taken", `${fmt(r.seconds, 1)}s`),
        kv("Next review", fmtDate(r.next_due))),
      h("div", { class: "row" }, next, state.item.game_id ? h("a", { href: `#games/${state.item.game_id}`, class: "small" }, "Review the game this came from") : null));
  }

  function nothingDue(stats) {
    const total = Object.values(stats.by_category || {}).reduce((a, v) => a + (num(v.total) || 0), 0);
    return [
      h("h2", null, total ? "Nothing due right now" : "No training positions yet"),
      h("p", { class: "lead" }, total
        ? `You're up to date${state.category ? " in this category" : ""}. Come back later, or build positions from newly analysed games.`
        : "Build exercises from the mistakes in your analysed games to start training."),
      h("div", null, buildButton(true)),
    ];
  }

  // -------------------------------------------------------------- stats & build
  function fillCategories(categories) {
    if (categorySelect.options.length > 1) return;
    for (const [key, label] of Object.entries(categories)) categorySelect.append(h("option", { value: key }, label));
    categorySelect.value = state.category;
  }

  function renderStats(stats) {
    const cats = Object.entries(stats.by_category || {});
    statsCard.replaceChildren(
      h("div", { class: "card-head" }, h("h2", null, "Your training")),
      cats.length
        ? h("div", { class: "table-wrap" }, h("table", { class: "data" },
          h("thead", null, h("tr", null, h("th", null, "Category"), h("th", { class: "r" }, "Due"), h("th", { class: "r" }, "Total"), h("th", { class: "r" }, "Solved at least once"))),
          h("tbody", null, cats.map(([name, v]) => h("tr", null,
            h("td", null, name), h("td", { class: "r num" }, fmt(v.due || 0)), h("td", { class: "r num" }, fmt(v.total)), h("td", { class: "r num" }, fmt(v.solved_later || 0)))))))
        : emptyNote("No positions yet."),
      h("p", { class: "row small" },
        `Failed at first, solved later: ${fmt(stats.failed_then_solved || 0)} · Reviews: ${fmt(stats.reviews || 0)} (${fmt(stats.correct || 0)} correct${stats.reviews ? `, ${pct(stats.correct / stats.reviews)}` : ""})`,
        tierBadge(FACT)));
  }

  function buildButton(primary = false) {
    const btn = h("button", { class: `btn ${primary ? "btn-primary" : ""}`, type: "button" }, h("span", { class: "spinner", "aria-hidden": "true" }), "Build new positions from my games");
    btn.addEventListener("click", async () => {
      btn.disabled = true;
      btn.classList.add("busy");
      buildStatus.textContent = "Building…";
      try {
        const res = await api("/api/training/build", { method: "POST", body: {} });
        buildStatus.textContent = res.created ? `Created ${plural(res.created, "new position")}.` : "No new positions — every mistake already has an exercise.";
        if (!state.item || res.created) loadNext();
      } catch (err) {
        buildStatus.textContent = `Build failed: ${err.message}`;
      } finally {
        btn.disabled = false;
        btn.classList.remove("busy");
      }
    });
    return btn;
  }
}

// ================================================================== Mirror & Elo
function renderMirror() {
  return h("div", { class: "view" },
    pageHead("Mirror & Elo", "How well the Mirror model predicts your moves on games it never saw, and what your results say about your strength."),
    card({ title: "Model versions" }, asyncBlock(() => api("/api/models"), modelsBlock, "Loading models…")),
    card({ title: "Last engine decision" }, asyncBlock(() => api("/api/last-decision"), lastDecisionBlock)),
    asyncBlock(() => api("/api/elo"), eloBlocks, "Estimating strength…"),
    card({ title: "Recorded games waiting for a result" }, asyncBlock(() => api("/api/pending"), pendingBlock)));
}

function modelsBlock(models) {
  if (!Array.isArray(models) || !models.length) {
    return emptyNote("No Mirror model yet. Train one with `mirror train` — if one is training now, reopen this tab when it finishes.");
  }
  const active = models.find((m) => m.is_active) || models[models.length - 1];
  const metrics = active.metrics || {};
  const names = Object.keys(metrics).filter((k) => !k.startsWith("_"));
  const sig = metrics._significance || {};
  return [
    h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", null, h("tr", null,
        h("th", null, "Version"), h("th", { class: "hide-sm" }, "Created"), h("th", { class: "r" }, "Top-1"), h("th", { class: "r" }, "Top-3"),
        h("th", { class: "r" }, "Log loss"), h("th", { class: "r hide-sm" }, "Test moves"),
        h("th", { class: "r hide-sm", title: "Share of moves losing 15%+ win chance: your real moves vs moves sampled from the model" }, "Blunder rate you / model"),
        h("th", { class: "r hide-sm", title: "Mean absolute difference in style rates (captures, checks, trades…) between you and the model" }, "Style error"))),
      h("tbody", null, models.slice().reverse().map((m) => {
        const mm = (m.metrics || {})[mirrorKey(m.metrics)] || {};
        const b = mm.behaviour || {};
        return h("tr", { class: m.is_active ? "highlight" : "" },
          h("td", null, h("strong", null, m.version), m.is_active ? [" ", chip("active", "chip-accent")] : null,
            m.notes ? h("span", { class: "sub" }, m.notes) : null),
          h("td", { class: "hide-sm num" }, fmtDate(m.created_at)),
          h("td", { class: "r num" }, pct(mm.top1, 1)),
          h("td", { class: "r num" }, pct(mm.top3, 1)),
          h("td", { class: "r num" }, fmt(mm.log_loss, 3)),
          h("td", { class: "r num hide-sm" }, fmt(mm.n)),
          h("td", { class: "r num hide-sm" }, ok(b.actual_blunder_rate) ? `${pct(b.actual_blunder_rate, 1)} / ${pct(b.model_blunder_rate, 1)}` : "–"),
          h("td", { class: "r num hide-sm" }, fmt(b.style_abs_error, 3)));
      })))),
    h("h3", null, `Baselines for ${active.version}${active.is_active ? " (active)" : ""}`),
    h("p", { class: "note" }, "Every model is scored on the same held-out moves. Top-1/top-3: how often your actual move was the model's first / top-three guess. Log loss: how surprised the model was by your move (lower is better)."),
    names.length ? metricsTable(metrics, names, { describe: true }) : emptyNote("This version has no held-out evaluation."),
    h("div", { class: "stack" },
      significance(sig.mirror_vs_population, "the population model (similar-rated opponents)"),
      significance(sig.mirror_vs_my_strength, "a strength-only model of you (no style)"),
      significance(sig.mirror_vs_generic_strength, "a generic Elo-matched strength model")),
  ];
}

function lastDecisionBlock(d) {
  if (!d || !d.fen) return emptyNote("No engine decision recorded yet. One appears after the Mirror engine plays a move in your chess GUI.");
  const ex = d.explanation || {};
  const moves = ex.moves || [];
  const orientation = sideToMove(d.fen);
  const when = ok(d.time) ? new Date(d.time * 1000).toLocaleString() : null;
  return h("div", { class: "decision" },
    h("div", { class: "stack" },
      h("img", { class: "board-img", src: boardUrl(d.fen, { orientation, arrow: d.chosen }), alt: "Position of the last engine decision", width: 480, height: 480 }),
      h("p", { class: "note" }, `Arrow: move played (${d.chosen}).`)),
    h("div", { class: "stack" },
      h("p", { class: "small muted-text" }, [d.model, when, ok(d.win_prob) ? `win chance ${pct(d.win_prob)}` : null].filter(Boolean).join(" · ")),
      moves.length ? h("ul", { class: "item-list" }, moves.map((m) => h("li", { style: "display:block" },
        h("div", { class: "spread" },
          h("strong", null, m.san || m.uci, m.uci === d.chosen ? h("span", { class: "muted-text" }, " (played)") : null),
          h("span", { class: "row small num" }, shareBar(m.p), pct(m.p, 1), m.book_count ? chip(`book ×${m.book_count}`) : null)),
        m.reasons && m.reasons.length ? h("ul", { class: "reasons" }, m.reasons.map((r) => h("li", null,
          h("span", null, r.text),
          TIERS[r.evidence] ? tierBadge(r.evidence) : h("span", { class: "badge tier-insufficient", title: "No clear personal difference from similar-rated players" }, "population behaviour")))) : null)))
        : emptyNote("No candidate moves in the explanation."),
      ok(ex.other_p) ? h("p", { class: "note" }, `All other moves together: ${pct(ex.other_p, 1)}.`) : null,
      (ex.context || []).map((c) => h("p", { class: "note" }, c)),
      ex.personalization ? h("p", { class: "note" }, ex.personalization) : null));
}

function eloBlocks(elo) {
  const results = elo.results || {};
  const quality = elo.quality || {};
  const combined = elo.combined || {};
  const pools = [...new Set([...Object.keys(results), ...Object.keys(quality), ...Object.keys(combined)])];
  pools.sort((a, b) => (a === elo.primary_pool ? -1 : b === elo.primary_pool ? 1 : a.localeCompare(b)));
  return [
    pools.length
      ? pools.map((pool) => poolCard(pool, results[pool], quality[pool], combined[pool], pool === elo.primary_pool))
      : card({ title: "Elo estimate" }, emptyNote("Not enough rated games yet (needs at least 5 in one pool).")),
    eloTestCard(elo.elotest),
  ];
}

function estimateFig(label, est, { tier, sub, muted } = {}) {
  return h("div", { class: `fig${muted ? " is-muted" : ""}` },
    h("span", { class: "label" }, label, tierBadge(tier)),
    h("span", { class: "value num" }, est && ok(est.mean) ? fmt(est.mean) : "–"),
    est && ok(est.lo) ? h("span", { class: "sub num" }, `95% interval ${range(est.lo, est.hi)}`) : null,
    sub ? h("span", { class: "sub" }, sub) : null);
}

function poolCard(pool, res, qual, comb, primary) {
  const figs = [];
  if (res) {
    figs.push(estimateFig("Results-based estimate", res.current, { tier: INFERENCE, sub: plural(res.n_games, "game") }));
    figs.push(estimateFig("Recent performance", res.recent_performance_30, { sub: `last ${plural(res.recent_performance_30 && res.recent_performance_30.n_games, "game")}` }));
    figs.push(h("div", { class: "fig" },
      h("span", { class: "label" }, "Platform-reported rating", tierBadge(FACT)),
      h("span", { class: "value num" }, ok(res.platform_reported_latest) ? fmt(res.platform_reported_latest) : "–"),
      h("span", { class: "sub" }, "latest game's rating header")));
  }
  if (qual && qual.status === "ok") {
    const cal = qual.calibration || {};
    figs.push(estimateFig("Quality-based estimate", qual, { tier: INFERENCE, sub: `calibrated on ${plural(cal.opponent_games, "opponent game")}` }));
  } else {
    figs.push(h("div", { class: "fig is-muted" },
      h("span", { class: "label" }, "Quality-based estimate", tierBadge(INSUFFICIENT)),
      h("span", { class: "value" }, qual && qual.status ? capitalize(qual.status) : "Needs 30+ rated opponent games in this pool")));
  }
  if (comb) figs.push(estimateFig("Combined", comb, { sub: "results + move quality (approximate)" }));

  return card({ title: `Elo · ${poolName(pool)}`, sub: primary ? "primary pool" : null },
    h("div", { class: "pool-figs" }, figs),
    res && res.trace ? [
      h("h3", { class: "small" }, "Results-based rating over time (mean with 95% band)"),
      ratingChart(res.trace, ok(res.platform_reported_latest) ? { reference: res.platform_reported_latest, referenceLabel: `platform ${fmt(res.platform_reported_latest)}` } : {}),
    ] : null);
}

function eloTestCard(test) {
  const message = h("div", { class: "message", hidden: true, role: "status" });
  const opponent = h("input", { type: "number", min: 1320, max: 3190, step: 10, placeholder: "auto", "aria-label": "Force opponent Elo (optional)", style: "width:110px" });
  const button = h("button", { class: "btn", type: "button" }, h("span", { class: "spinner", "aria-hidden": "true" }), "Choose next Elo-test opponent");
  button.addEventListener("click", async () => {
    button.disabled = true;
    button.classList.add("busy");
    try {
      const body = ok(opponent.value) && opponent.value !== "" ? { opponent: Number(opponent.value) } : {};
      const res = await api("/api/elotest/next", { method: "POST", body });
      message.textContent = res.message || "Done.";
    } catch (err) {
      message.textContent = `Could not choose an opponent: ${err.message}`;
    } finally {
      message.hidden = false;
      button.disabled = false;
      button.classList.remove("busy");
    }
  });
  return card({ title: "Elo test (vs Stockfish UCI_LimitStrength)" },
    test && test.current
      ? [h("div", { class: "pool-figs" }, estimateFig("Elo-test estimate", test.current, { tier: INFERENCE, sub: plural(test.current.n_games, "game") })),
        h("p", { class: "note" }, test.scale || ""),
        test.trace && test.trace.length > 1 ? ratingChart(test.trace) : null]
      : emptyNote("No Elo-test games yet. Pick an opponent, set the engine's Mode to EloTest and play it in your GUI."),
    h("div", { class: "inline-form" }, h("label", { class: "row small" }, "Opponent Elo", opponent), button),
    message);
}

function pendingBlock(items) {
  if (!Array.isArray(items) || !items.length) return emptyNote("No recorded games waiting for a result.");
  return h("div", { class: "pending-list" }, items.map((c) => {
    const status = h("span", { class: "note", role: "status" });
    const buttons = [["1-0", "1-0"], ["½-½", "1/2-1/2"], ["0-1", "0-1"]].map(([label, result]) =>
      h("button", {
        class: "btn btn-small", type: "button", title: `Record result ${result} (White's point of view)`,
        onclick: async (e) => {
          if (!window.confirm(`Record game ${c.id} as ${result}? It will be imported into your games.`)) return;
          const all = e.currentTarget.parentElement.querySelectorAll("button");
          all.forEach((b) => { b.disabled = true; });
          try {
            const res = await api("/api/pending/finalize", { method: "POST", body: { id: c.id, result } });
            status.textContent = res.message || "Recorded.";
          } catch (err) {
            status.textContent = `Failed: ${err.message}`;
            all.forEach((b) => { b.disabled = false; });
          }
        },
      }, label));
    const you = c.engine_color === "white" ? "black" : "white";
    return h("div", { class: "pending-item" },
      h("div", null,
        h("strong", null, c.id), " ",
        h("span", { class: "note" }, [fmtDate(c.started), c.mode || "Mirror", `you played ${you}`, plural((c.moves || []).length, "ply", "plies"),
          c.opponent_rating ? `opponent ${c.opponent_rating}` : c.model].filter(Boolean).join(" · "))),
      h("div", { class: "row" }, buttons, status));
  }));
}

// ================================================================== play against the bot
const TIME_CONTROLS = [
  { key: "untimed", base: null, inc: 0, label: "Untimed" },
  { key: "3+2", base: 180, inc: 2, label: "3 + 2 (blitz)" },
  { key: "5+3", base: 300, inc: 3, label: "5 + 3 (blitz)" },
  { key: "10+5", base: 600, inc: 5, label: "10 + 5 (rapid)" },
  { key: "15+10", base: 900, inc: 10, label: "15 + 10 (rapid)" },
  { key: "30+0", base: 1800, inc: 0, label: "30 + 0 (classical)" },
];
const OPPONENTS = {
  mirror: { title: "Your Mirror", text: "Plays the moves you would most likely play, sampled from your model. It becomes more like you with every game you finish." },
  elotest: { title: "Elo test", text: "Calibrated Stockfish (UCI_LimitStrength) at a rating chosen from your current estimate. Results update your Elo-test rating automatically." },
  stockfish: { title: "Sparring", text: "Stockfish limited to a rating you choose (1320–3190)." },
};

function renderPlay() {
  const body = h("div", { class: "async" }, loadingEl());
  api("/api/play/current")
    .then((st) => (st ? playGame(body, st) : playSetup(body)))
    .catch((err) => body.replaceChildren(errorBox(err, () => route())));
  return h("div", { class: "view" },
    pageHead("Play", "Play against your Mirror or a calibrated opponent. Every game you finish is saved as your data, and the Mirror learns from it automatically."),
    body);
}

function playSetup(body) {
  const pref = (() => { try { return JSON.parse(localStorage.getItem("play-prefs") || "{}"); } catch { return {}; } })();
  let opponent = pref.opponent || "mirror";
  const radios = Object.entries(OPPONENTS).map(([key, o]) => {
    const input = h("input", { type: "radio", name: "opp", value: key, checked: key === opponent });
    input.addEventListener("change", () => { opponent = key; sync(); });
    return h("label", { class: "opp-choice" }, input, h("span", null, h("strong", null, o.title), h("span", { class: "note" }, o.text)));
  });
  const elo = h("input", { type: "number", min: "1320", max: "3190", step: "10", value: pref.elo || 1500, style: "width:7em" });
  const eloRow = h("div", { class: "row" }, h("span", { class: "note" }, "Stockfish rating"), elo);
  const color = h("select", null, h("option", { value: "white" }, "White"), h("option", { value: "random" }, "Random"), h("option", { value: "black" }, "Black"));
  color.value = pref.color || "random";
  const tc = h("select", null, TIME_CONTROLS.map((t) => h("option", { value: t.key }, t.label)));
  tc.value = pref.tc || "5+3";
  const human = h("input", { type: "checkbox", checked: pref.human !== false });
  const humanRow = h("label", { class: "row small" }, human, "Mirror uses your own thinking-time habits (timed games)");
  const status = h("span", { class: "note", role: "status" });
  const start = h("button", { class: "btn btn-primary", type: "button" }, "Start game");
  function sync() {
    eloRow.hidden = opponent !== "stockfish";
    humanRow.hidden = opponent !== "mirror";
  }
  sync();
  start.addEventListener("click", async () => {
    const t = TIME_CONTROLS.find((x) => x.key === tc.value);
    const prefs = { opponent, elo: num(elo.value), color: color.value, tc: tc.value, human: human.checked };
    try { localStorage.setItem("play-prefs", JSON.stringify(prefs)); } catch { /* private mode */ }
    start.disabled = true;
    status.textContent = opponent === "mirror" ? "Setting up your Mirror…" : "Setting up the board…";
    try {
      const st = await api("/api/play/new", { method: "POST", body: { opponent, elo: prefs.elo, color: color.value, base: t.base, inc: t.inc, human_timing: human.checked } });
      try {
        playGame(body, st);
      } catch (err) {
        console.error(err);
        body.replaceChildren(errorBox(err, () => route()));
      }
    } catch (err) {
      status.textContent = err.message;
      start.disabled = false;
    }
  });
  body.replaceChildren(card({ title: "New game" },
    h("div", { class: "opp-choices" }, radios),
    eloRow,
    h("div", { class: "row" }, field("Your colour", color), field("Time control", tc)),
    humanRow,
    h("div", { class: "row" }, start, status),
    h("p", { class: "note" }, "Only your own moves are learned from. Your thinking time per move is recorded too, so the Mirror also learns how you play under time pressure.")));
}

function playGame(body, initial) {
  const g = { st: initial, legal: initial.legal || [], sel: null, targets: [], pending: null, geom: null, busy: false,
    drawToken: 0, blobUrl: null, receivedAt: performance.now(), timer: null, flagged: false, learnTimer: null };
  let statusEl = null;
  const img = h("img", { class: "board-img", alt: "Game board — click a piece, then its destination", width: 480, height: 480, draggable: "false" });
  const boardWrap = h("div", { class: "board-wrap" }, img);
  const oppClock = h("div", { class: "clock" });
  const myClock = h("div", { class: "clock" });
  const oppName = h("span", { class: "player-name" });
  const myName = h("span", { class: "player-name" });
  const boardCol = h("div", { class: "stack" },
    h("div", { class: "clock-row" }, oppName, oppClock), boardWrap, h("div", { class: "clock-row" }, myName, myClock));
  const panel = h("div", { class: "card stack play-panel" });
  body.replaceChildren(h("div", { class: "trainer" }, boardCol, panel));
  img.addEventListener("click", onClick);
  update(initial);

  function update(st) {
    g.st = st;
    g.legal = st.legal || [];
    g.sel = null;
    g.targets = [];
    g.pending = null;
    g.receivedAt = performance.now();
    g.flagged = false;
    oppName.textContent = st.opponent_label;
    myName.textContent = st.player_name || "You";
    boardWrap.classList.toggle("locked", !st.your_turn);
    drawBoard().catch((err) => panel.append(errorBox(err)));
    renderPanel();
    tick();
    clearInterval(g.timer);
    if (st.status === "playing" && st.clocks) g.timer = setInterval(tick, 200);
    if (st.status === "over" && st.game_id) watchLearning();
  }

  function clockValue(color) {
    const st = g.st;
    if (!st.clocks) return null;
    let v = st.clocks[color];
    if (st.status === "playing" && st.turn === color && !g.pending) v -= (performance.now() - g.receivedAt) / 1000;
    return Math.max(0, v);
  }

  function tick() {
    const st = g.st;
    if (!body.isConnected) { clearInterval(g.timer); return; }
    const me = st.orientation;
    const opp = me === "white" ? "black" : "white";
    for (const [el, color] of [[myClock, me], [oppClock, opp]]) {
      const v = clockValue(color);
      el.textContent = v == null ? "" : fmtClock(v);
      el.classList.toggle("running", st.status === "playing" && st.turn === color);
      el.classList.toggle("low", v != null && v < 20);
    }
    const mine = clockValue(me);
    if (mine === 0 && st.status === "playing" && st.your_turn && !g.flagged) {
      g.flagged = true;
      api("/api/play/flag", { method: "POST", body: { id: st.id } }).then(update).catch(() => {});
    }
  }

  async function drawBoard() {
    const st = g.st;
    const token = ++g.drawToken;
    const params = { orientation: st.orientation, sel: g.sel, targets: g.targets, last: g.pending || st.last };
    let url = boardUrl(st.fen, params);
    if (g.pending) url += `&apply=${g.pending}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`board image failed (${res.status})`);
    const text = await res.text();
    if (token !== g.drawToken) return;
    g.geom = parseBoardGeometry(text, st.orientation);
    const old = g.blobUrl;
    g.blobUrl = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
    img.src = g.blobUrl;
    if (old) setTimeout(() => URL.revokeObjectURL(old), 2000);
  }

  function onClick(ev) {
    const st = g.st;
    if (!st.your_turn || g.busy || !g.geom) return;
    const sq = squareAt(g.geom, img, ev.clientX, ev.clientY);
    if (!sq) return;
    if (g.sel && g.targets.includes(sq)) {
      const options = g.legal.filter((u) => u.startsWith(g.sel + sq));
      send(options.find((u) => u.length === 4) || options.find((u) => u.endsWith("q")) || options[0]);
      return;
    }
    const from = g.legal.filter((u) => u.startsWith(sq));
    if (from.length && sq !== g.sel) {
      g.sel = sq;
      g.targets = [...new Set(from.map((u) => u.slice(2, 4)))];
    } else {
      g.sel = null;
      g.targets = [];
    }
    drawBoard().catch(() => {});
  }

  async function send(uci) {
    if (!uci) return;
    g.busy = true;
    const myLeft = clockValue(g.st.orientation);
    g.pending = uci;
    g.sel = null;
    g.targets = [];
    if (g.st.clocks && myLeft != null) g.st.clocks[g.st.orientation] = myLeft + (g.st.inc || 0);
    boardWrap.classList.add("locked");
    drawBoard().catch(() => {});
    setStatus(g.st.opponent === "mirror" ? "Your Mirror is thinking…" : "Your opponent is thinking…", true);
    try {
      update(await api("/api/play/move", { method: "POST", body: { id: g.st.id, uci } }));
    } catch (err) {
      g.pending = null;
      boardWrap.classList.remove("locked");
      drawBoard().catch(() => {});
      setStatus(err.message);
    } finally {
      g.busy = false;
    }
  }

  function setStatus(text, spinning = false) {
    if (!statusEl) return;
    statusEl.replaceChildren(spinning ? h("span", { class: "spinner", "aria-hidden": "true" }) : "", h("span", null, text));
  }

  function renderPanel() {
    const st = g.st;
    statusEl = h("p", { class: "row play-status", role: "status" });
    const over = st.status === "over";
    const moves = h("ol", { class: "movelist" });
    for (let i = 0; i < st.moves.length; i += 2) {
      moves.append(h("li", null, h("span", { class: "mv" }, st.moves[i]), st.moves[i + 1] ? h("span", { class: "mv" }, st.moves[i + 1]) : ""));
    }
    const actions = h("div", { class: "row" });
    if (!over) {
      if (st.moves.length < 2) actions.append(h("button", { class: "btn btn-small", type: "button", onclick: () => act("/api/play/abort") }, "Abort"));
      actions.append(
        h("button", { class: "btn btn-small", type: "button", onclick: () => act("/api/play/draw") }, "Offer draw"),
        h("button", { class: "btn btn-small", type: "button", onclick: () => { if (confirm("Resign this game?")) act("/api/play/resign"); } }, "Resign"));
    } else {
      actions.append(h("button", { class: "btn btn-primary", type: "button", onclick: () => playSetup(body) }, "New game"));
      if (st.game_id) actions.append(h("a", { class: "btn", href: `#games/${st.game_id}` }, "Review this game"));
    }
    panel.replaceChildren(...[
      h("div", { class: "row" }, chip(OPPONENTS[st.opponent] ? OPPONENTS[st.opponent].title : st.opponent, "chip-accent"),
        st.base ? chip(`${Math.round(st.base / 60)} + ${st.inc}`) : chip("untimed"), chip(`you: ${st.orientation}`)),
      over ? resultBanner(st) : statusEl,
      actions,
      st.moves.length ? moves : h("p", { class: "note" }, st.your_turn ? "You move first." : ""),
      st.bot && st.bot.predicted ? thinkingBlock(st.bot) : null,
      over ? h("div", { class: "learn", id: "learn-status" }) : null].filter(Boolean));
    if (!over) setStatus(st.your_turn ? (st.check ? "Check! Your move." : "Your move.") : "Waiting for your opponent…");
  }

  function resultBanner(st) {
    const me = st.orientation === "white" ? "1-0" : "0-1";
    const outcome = st.result === "1/2-1/2" ? "Draw" : st.result === me ? "You won" : st.result === "*" ? "Game ended" : "You lost";
    return h("div", { class: `result-banner ${outcome === "You won" ? "good" : outcome === "You lost" ? "bad" : ""}` },
      h("strong", null, outcome), h("span", null, ` · ${st.termination || ""} · ${st.result}`),
      st.draw_offer === "declined" ? h("span", { class: "note" }, " (draw declined)") : null,
      st.save_error ? h("p", { class: "form-error" }, `Could not save the game: ${st.save_error}`) : null);
  }

  function thinkingBlock(bot) {
    const d = h("details", { class: "thinking" },
      h("summary", null, `What your Mirror considered before ${bot.chosen}`),
      h("ul", { class: "plain" }, bot.predicted.map((m) =>
        h("li", null, h("strong", null, m.san), ` ${pct(m.p, 0)}`, m.book ? h("span", { class: "note" }, ` · you played it ${m.book}× here`) : ""))),
      h("p", { class: "note" }, `Model ${bot.model}. The move is sampled from these probabilities, not picked by Stockfish.`));
    return d;
  }

  async function act(path) {
    try {
      const st = await api(path, { method: "POST", body: { id: g.st.id } });
      update(st);
      if (st.draw_offer === "declined") setStatus("Draw declined. Your move.");
    } catch (err) {
      setStatus(err.message);
    }
  }

  function watchLearning() {
    clearTimeout(g.learnTimer);
    const box = () => document.getElementById("learn-status");
    const poll = async () => {
      const el = box();
      if (!el) return;
      const jobs = await api("/api/jobs").catch(() => []);
      const job = jobs.find((j) => j.kind.startsWith("learning"));
      if (!job) {
        el.replaceChildren(h("p", { class: "note" }, `Saved as game #${g.st.game_id}.`));
      } else if (job.status === "running") {
        el.replaceChildren(h("p", { class: "row note" }, h("span", { class: "spinner", "aria-hidden": "true" }),
          `Saved as game #${g.st.game_id}. Your Mirror is learning from it… ${(job.log || []).slice(-1)[0] || ""}`));
        g.learnTimer = setTimeout(poll, 1500);
        return;
      } else {
        el.replaceChildren(h("p", { class: "note" }, job.status === "done"
          ? `Saved as game #${g.st.game_id}. ${(job.log || []).slice(-2).join(" ")}`
          : `Saved as game #${g.st.game_id}, but learning failed: ${(job.log || []).slice(-1)[0] || ""}`));
      }
    };
    setTimeout(poll, 400);
  }
}

// ================================================================== setup: identity, import, update, engine
const nativeApi = () => (window.pywebview && window.pywebview.api) || null;

function renderSetup() {
  return h("div", { class: "view" },
    pageHead("Setup", "Who you are, where your games come from, and keeping your Mirror up to date."),
    asyncBlock(() => api("/api/settings"), setupBlocks, "Loading settings…"),
    card({ title: "Activity", sub: "background jobs" }, jobsBlock()));
}

function setupBlocks(st) {
  const firstRun = !st.games;
  return [
    firstRun && h("div", { class: "message welcome" },
      h("strong", null, "Welcome. "),
      "Either import your online games (enter your usernames, import, then press ",
      h("em", null, "Update everything"), ") or simply ", h("a", { href: "#play" }, "start playing your Mirror"),
      " — it learns from every game you finish. Only your own moves are learned from."),
    h("div", { class: "grid-2" }, identityCard(st), importCard(st)),
    h("div", { class: "grid-2" }, updateCard(st), engineCard(st)),
  ];
}

function field(label, input, hint) {
  return h("label", { class: "field" }, h("span", { class: "field-label" }, label), input, hint && h("span", { class: "note" }, hint));
}

function identityCard(st) {
  const p = st.player || {};
  const name = h("input", { type: "text", value: p.name || "", placeholder: "e.g. Ahmed" });
  const aliases = h("input", { type: "text", value: (p.aliases || []).join(", "), placeholder: "other names, comma separated" });
  const lichess = h("input", { type: "text", value: p.lichess || "", placeholder: "Lichess username" });
  const chesscom = h("input", { type: "text", value: p.chesscom || "", placeholder: "Chess.com username" });
  const rating = h("input", { type: "number", min: "400", max: "3000", step: "50", value: p.rating || 1500 });
  const status = h("span", { class: "note", role: "status" });
  const save = h("button", { class: "btn btn-primary", type: "button" }, "Save");
  save.addEventListener("click", async () => {
    save.disabled = true;
    try {
      await api("/api/settings", { method: "POST", body: { name: name.value, aliases: aliases.value, lichess: lichess.value, chesscom: chesscom.value, rating: num(rating.value) || null } });
      status.textContent = "Saved.";
      document.getElementById("player-name").textContent = name.value;
    } catch (err) {
      status.textContent = `Could not save: ${err.message}`;
    } finally {
      save.disabled = false;
    }
  });
  return card({ title: "You" },
    h("div", { class: "form" },
      field("Name in your PGN files", name, "Games are matched to you by these names (case-insensitive)."),
      field("Other names", aliases),
      h("div", { class: "grid-2 tight" }, field("Lichess", lichess), field("Chess.com", chesscom)),
      field("Approximate rating", rating, "Only a starting point for the Mirror until it has seen enough of your games.")),
    h("div", { class: "row" }, save, status),
    h("dl", { class: "kv" },
      kv("Games imported", `${fmt(st.games)} (${fmt(st.my_games)} with you, ${fmt(st.analyzed)} analysed)`),
      kv("Model versions", fmt(st.models)),
      kv("Data folder", h("span", { class: "row" }, h("code", null, st.data_dir),
        nativeApi() && h("button", { class: "btn btn-small", type: "button", onclick: () => nativeApi().reveal_data_dir() }, "Show in Finder")))));
}

async function startJob(path, body, status) {
  try {
    const job = await api(path, { method: "POST", body });
    status.textContent = `Started: ${job.kind}. Progress is shown under Activity.`;
    pollJobs();
  } catch (err) {
    status.textContent = err.message;
  }
}

function importCard(st) {
  const p = st.player || {};
  const status = h("span", { class: "note", role: "status" });
  const maxGames = h("input", { type: "number", min: "1", placeholder: "all", style: "width:7em" });
  const months = h("input", { type: "number", min: "1", placeholder: "all", style: "width:7em" });
  const lichessBtn = h("button", { class: "btn", type: "button" }, "Download from Lichess");
  lichessBtn.addEventListener("click", () => startJob("/api/import/fetch", { site: "lichess", user: p.lichess, max: num(maxGames.value) || null }, status));
  const chessBtn = h("button", { class: "btn", type: "button" }, "Download from Chess.com");
  chessBtn.addEventListener("click", () => startJob("/api/import/fetch", { site: "chesscom", user: p.chesscom, months: num(months.value) || null }, status));
  const fileInput = h("input", { type: "file", accept: ".pgn,text/plain", multiple: true, hidden: true });
  fileInput.addEventListener("change", async () => {
    for (const file of fileInput.files) {
      await startJob("/api/import/text", { filename: file.name, text: await file.text() }, status);
      await waitForIdle();
    }
    fileInput.value = "";
  });
  const pgnBtn = h("button", { class: "btn", type: "button" }, "Import PGN files…");
  pgnBtn.addEventListener("click", async () => {
    const native = nativeApi();
    if (native && native.pick_pgn_files) {
      const paths = await native.pick_pgn_files();
      if (paths && paths.length) startJob("/api/import/paths", { paths }, status);
    } else {
      fileInput.click();
    }
  });
  return card({ title: "Import games" },
    h("p", { class: "note" }, "Games are kept verbatim; importing the same game twice is detected. 20% of games are held out permanently to measure how well the Mirror predicts you."),
    h("div", { class: "row" }, lichessBtn, h("span", { class: "note" }, "max games"), maxGames),
    h("div", { class: "row" }, chessBtn, h("span", { class: "note" }, "last months"), months),
    h("div", { class: "row" }, pgnBtn, fileInput),
    status);
}

function updateCard(st) {
  const status = h("span", { class: "note", role: "status" });
  const depth = h("select", null,
    h("option", { value: "10" }, "Quick (depth 10)"),
    h("option", { value: "12", selected: true }, "Standard (depth 12)"),
    h("option", { value: "14" }, "Thorough (depth 14)"));
  depth.value = String((st.analysis && st.analysis.deep_depth) || 12);
  const limit = h("input", { type: "number", min: "1", placeholder: "all", style: "width:7em" });
  const run = (steps) => startJob("/api/jobs/pipeline", { steps, depth: num(depth.value), limit: num(limit.value) || null }, status);
  const all = h("button", { class: "btn btn-primary", type: "button" }, "Update everything");
  all.addEventListener("click", () => run(["analyze", "train", "trainer"]));
  const analyzeOnly = h("button", { class: "btn btn-small", type: "button" }, "Analyse only");
  analyzeOnly.addEventListener("click", () => run(["analyze"]));
  const trainOnly = h("button", { class: "btn btn-small", type: "button" }, "Train only");
  trainOnly.addEventListener("click", () => run(["train", "trainer"]));
  const workers = (st.analysis && st.analysis.workers) || 8;
  const perPos = { 10: 0.2, 12: 0.45, 14: 1.1 };
  const estimate = () => {
    const games = Math.max(0, (st.games || 0) - (st.analyzed || 0));
    const n = Math.min(games, num(limit.value) || games);
    const secs = (n * 80 * (perPos[depth.value] || 0.5)) / workers;
    return n ? `≈ ${Math.max(1, Math.round(secs / 60))} min for ${plural(n, "new game")} on ${workers} cores` : "No unanalysed games.";
  };
  const est = h("span", { class: "note" }, estimate());
  depth.addEventListener("change", () => { est.textContent = estimate(); });
  limit.addEventListener("input", () => { est.textContent = estimate(); });
  return card({ title: "Update your Mirror" },
    h("p", { class: "note" }, "Runs Stockfish over new games (yours and your opponents'), trains a new model version, evaluates it on held-out games, and turns your new mistakes into training positions."),
    h("div", { class: "row" }, field("Analysis depth", depth), field("Only the most recent", limit, "games (blank = all)")),
    est,
    h("div", { class: "row" }, all, analyzeOnly, trainOnly),
    status);
}

function engineCard(st) {
  const ec = st.encroissant || {};
  const status = h("span", { class: "note", role: "status" });
  const add = h("button", { class: "btn", type: "button", disabled: !ec.installed || ec.registered || ec.running }, ec.registered ? "Added to En Croissant" : "Add to En Croissant");
  add.addEventListener("click", async () => {
    add.disabled = true;
    try {
      const r = await api("/api/engine/install", { method: "POST", body: {} });
      status.textContent = r.message;
    } catch (err) {
      status.textContent = err.message;
      add.disabled = false;
    }
  });
  const path = h("code", { class: "path" }, st.engine_path);
  const copy = h("button", { class: "btn btn-small", type: "button" }, "Copy path");
  copy.addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(st.engine_path); copy.textContent = "Copied"; } catch { copy.textContent = "Select and copy the path"; }
  });
  return card({ title: "Play it in En Croissant" },
    h("p", { class: "note" }, "The Mirror is a standard UCI engine. In En Croissant: Engines → Add engine → Local, and choose this file:"),
    h("div", { class: "row" }, path, copy),
    h("div", { class: "row" }, add,
      !ec.installed && h("span", { class: "note" }, "En Croissant not found in /Applications."),
      ec.running && !ec.registered && h("span", { class: "note" }, "Quit En Croissant first — it rewrites its engine list on exit.")),
    h("p", { class: "note" }, "Then play the engine in Play (Mode = Mirror plays like you; Mode = EloTest measures your strength). In analysis, its lines are the moves you would most likely play. Keep En Croissant's own opening book off."),
    h("dl", { class: "kv" }, kv("Stockfish", st.stockfish || "not found")),
    status);
}

// ---- background jobs
let jobsTimer = null;
const jobsListeners = new Set();

async function pollJobs() {
  clearTimeout(jobsTimer);
  let jobs = [];
  try { jobs = await api("/api/jobs"); } catch { jobs = []; }
  for (const fn of jobsListeners) fn(jobs);
  if (jobs.some((j) => j.status === "running")) jobsTimer = setTimeout(pollJobs, 1500);
}

function waitForIdle() {
  return new Promise((resolve) => {
    const check = async () => {
      const jobs = await api("/api/jobs").catch(() => []);
      if (jobs.some((j) => j.status === "running")) setTimeout(check, 1000);
      else resolve();
    };
    check();
  });
}

function jobsBlock() {
  const box = h("div", { class: "jobs" }, emptyNote("Nothing has run yet in this session."));
  const seen = new Map();
  const render = (jobs) => {
    if (!box.isConnected && seen.size) { jobsListeners.delete(render); return; }
    if (!jobs.length) return;
    box.replaceChildren(...jobs.map((j) => {
      const secs = ((j.ended || Date.now() / 1000) - j.started);
      const chipCls = j.status === "done" ? "chip-good" : j.status === "error" ? "chip-bad" : "";
      const finished = seen.get(j.id) === "running" && j.status !== "running";
      seen.set(j.id, j.status);
      return h("div", { class: "job" },
        h("div", { class: "row" }, j.status === "running" && h("span", { class: "spinner", "aria-hidden": "true" }),
          h("strong", null, capitalize(j.kind)), chip(j.status, chipCls), h("span", { class: "note" }, `${Math.round(secs)}s`),
          finished && j.status === "done" && h("a", { href: "#overview", class: "small" }, "See updated results →")),
        h("pre", { class: "log" }, (j.log || []).slice(-14).join("\n") || "…"));
    }));
  };
  jobsListeners.add(render);
  pollJobs();
  return box;
}

// ================================================================== router & shell
const ROUTES = {
  overview: { title: "Overview", render: renderOverview },
  play: { title: "Play", render: renderPlay },
  weaknesses: { title: "Weaknesses", render: renderWeaknesses },
  openings: { title: "Openings", render: renderOpenings },
  games: { title: "Games & Coach", render: renderGames },
  training: { title: "Training", render: renderTraining },
  mirror: { title: "Mirror & Elo", render: renderMirror },
  setup: { title: "Setup", render: renderSetup },
};

const main = document.getElementById("main");

function setPlayerName(summary) {
  if (summary && summary.player) document.getElementById("player-name").textContent = summary.player;
}
let currentKey = null;

function route() {
  const [name, arg] = location.hash.replace(/^#\/?/, "").split("/");
  const key = ROUTES[name] ? name : "overview";
  for (const a of document.querySelectorAll(".tabs a")) {
    if (a.dataset.route !== key) {
      a.removeAttribute("aria-current");
      continue;
    }
    a.setAttribute("aria-current", "page");
    const nav = a.parentElement; // keep the active tab visible in the scrolling nav on phones
    if (a.offsetLeft < nav.scrollLeft || a.offsetLeft + a.offsetWidth > nav.scrollLeft + nav.clientWidth) {
      nav.scrollLeft = a.offsetLeft - 16;
    }
  }
  document.title = `${ROUTES[key].title} · Chess Mirror`;
  try {
    main.replaceChildren(ROUTES[key].render(arg ? decodeURIComponent(arg) : undefined));
  } catch (err) {
    console.error(err);
    main.replaceChildren(errorBox(err, route));
  }
  const routeId = `${key}/${arg || ""}`;
  if (routeId !== currentKey) window.scrollTo(0, 0);
  currentKey = routeId;
}

const REFRESH_STEPS = [
  ["/api/summary", "profile"],
  ["/api/weaknesses", "weaknesses"],
  ["/api/openings", "openings"],
  ["/api/elo", "Elo estimates"],
];

async function refreshAnalysis(button, status) {
  button.disabled = true;
  button.classList.add("busy");
  try {
    for (const [path, label] of REFRESH_STEPS) {
      status.textContent = `Recomputing ${label}… this can take a minute.`;
      await api(path, { refresh: true });
    }
    status.textContent = "Analysis refreshed.";
    route();
    setTimeout(() => { if (status.textContent === "Analysis refreshed.") status.textContent = ""; }, 4000);
  } catch (err) {
    status.textContent = `Refresh failed: ${err.message}`;
  } finally {
    button.disabled = false;
    button.classList.remove("busy");
  }
}

function init() {
  const button = document.getElementById("refresh");
  const status = document.getElementById("status");
  button.addEventListener("click", () => refreshAnalysis(button, status));
  window.addEventListener("hashchange", route);
  const startsOnOverview = !ROUTES[location.hash.replace(/^#\/?/, "").split("/")[0]] || location.hash.startsWith("#overview");
  // First run (no games yet): start on Setup instead of an empty overview.
  api("/api/settings").then((st) => {
    if (!st.games && (!location.hash || location.hash === "#overview")) location.hash = "#setup";
    else if (st.player && st.player.name) document.getElementById("player-name").textContent = st.player.name;
  }).catch(() => {});
  route();
  // The overview fetches the summary itself; elsewhere fetch it once for the player's name.
  // (Never fire two summary requests at once: the server would compute the profile twice.)
  if (!startsOnOverview) api("/api/summary").then(setPlayerName).catch(() => {});
}

init();
