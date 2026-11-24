"""M3: generate an offline Plotly explorer and a reproducible PNG snapshot.

Reuse M1's bands and direct gap and M2's FHS results. Moving the mass slider
must update the two panels together and expose both valley gap closures.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.io import to_html

from haldane import (
    CRITICAL_MASS, GAP_GRID_SIZE, K, MASS_MAX, MASS_MIN, PHI, T1, T2,
    brillouin_zone_grid, hamiltonian, minimum_direct_gap, symmetry_path,
)
from m2_chern import GRID_SIZE, band_chern_numbers


LOWER = "#2459a7"
UPPER = "#b94d26"
INK = "#202b3b"
GRID = "#e7ecf4"
AXES = dict(zerolinecolor=GRID, gridcolor=GRID, fixedrange=True, automargin=True)
PLOT_CONFIG = dict(responsive=True, displayModeBar=False, displaylogo=False)


def build_samples(path, valley_path):
    regular = np.linspace(MASS_MIN, MASS_MAX, 601).round(6)
    masses = np.sort(np.r_[regular, -CRITICAL_MASS, CRITICAL_MASS])
    momenta = brillouin_zone_grid(GAP_GRID_SIZE)
    hopping = dict(t1=T1, t2=T2, phi=PHI)
    points = np.vstack([path, valley_path])
    samples = []
    for mass in masses:
        eigenvalues = np.linalg.eigvalsh(hamiltonian(points, mass=mass, **hopping))
        bands = eigenvalues.round(7).T
        gap = minimum_direct_gap(mass, momenta, **hopping)
        chern = band_chern_numbers(mass, GRID_SIZE)
        samples.append({
            "mass": float(mass),
            "gap": round(float(gap), 12),
            "bands": [*bands[:, :len(path)].tolist(), *bands[:, len(path):].tolist()],
            "chern": [None if np.isnan(value) else round(value) for value in chern],
        })
    return samples


def band_figure(distance, ticks, valley_offsets, initial):
    figure = go.Figure()
    for index, energies in enumerate(initial["bands"]):
        valley = index >= 2
        upper = index % 2 == 1
        figure.add_scatter(
            x=valley_offsets if valley else distance,
            y=energies,
            xaxis="x2" if valley else "x",
            name="上能带" if upper else "下能带",
            mode="lines",
            line=dict(color=UPPER if upper else LOWER, width=2.6),
            hovertemplate="E/t₁ = %{y:.5f}<extra>%{fullData.name}</extra>",
        )
    figure.update_layout(
        xaxis=dict(**AXES, domain=[0, 0.74], tickvals=ticks, ticktext=["Γ", "K", "M", "Γ"],
                   title_text="Γ–K–M–Γ", range=[0, float(distance[-1])]),
        xaxis2=dict(**AXES, domain=[0.81, 1], tickvals=[0], ticktext=["K′"],
                    title_text="K′ 附近", range=[-0.28, 0.28]),
        yaxis=dict(**AXES, title_text="能量 E/t₁", range=[-3.5, 3.5]),
        showlegend=False,
    )
    return figure


def chern_figure(samples, initial):
    defined = [sample for sample in samples if sample["chern"][0] is not None]
    masses = [sample["mass"] for sample in defined]
    figure = go.Figure()
    for band, (color, name) in enumerate([(LOWER, "下带"), (UPPER, "上带")]):
        figure.add_scatter(
            x=masses, y=[sample["chern"][band] for sample in defined], mode="lines",
            line=dict(color=color, width=2.8, shape="hvh", dash="solid" if band == 0 else "dash"),
            name=name, hoverinfo="skip",
        )
    for mass in (-CRITICAL_MASS, CRITICAL_MASS):
        figure.add_vline(x=mass, line=dict(color="#8793a5", dash="dash", width=1.5))
    figure.add_scatter(x=[initial["mass"]] * 2, y=[-1.32, 1.18], mode="lines",
                       line=dict(color=INK, width=1.4), hoverinfo="skip")
    figure.add_scatter(
        x=[initial["mass"]] * 2, y=initial["chern"], mode="markers", hoverinfo="skip",
        marker=dict(size=[13, 8], color=[LOWER, UPPER], symbol=["circle-open", "circle"],
                    line=dict(width=2)),
    )
    figure.update_layout(
        xaxis=dict(**AXES, range=[MASS_MIN - 0.05, MASS_MAX + 0.05], title_text="质量项 M/t₁"),
        yaxis=dict(**AXES, range=[-1.32, 1.32], title_text="陈数 C", tickvals=[-1, 0, 1]),
        showlegend=False,
    )
    return figure


def plot_embed(figure, div_id, *, include_plotlyjs):
    figure.update_layout(
        height=350, margin=dict(l=54, r=24, t=14, b=52),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Segoe UI, Microsoft YaHei, sans-serif", size=12, color=INK),
    )
    return to_html(figure, full_html=False, include_plotlyjs=include_plotlyjs,
                   config=PLOT_CONFIG, div_id=div_id)


def save_snapshot(bands, chern, sample):
    """Render the explorer's data at one mass without a browser dependency."""
    import matplotlib.pyplot as plt

    figure, (band_axis, chern_axis) = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    for index, color in enumerate((LOWER, UPPER)):
        band_axis.plot(bands.data[index].x, sample["bands"][index], color=color, lw=2)
        phase = chern.data[index]
        chern_axis.step(phase.x, phase.y, where="mid", color=color, lw=2,
                        ls="-" if index == 0 else "--", label=("Lower band", "Upper band")[index])
        chern_axis.scatter(sample["mass"], sample["chern"][index], color=color, zorder=4)
    band_axis.set_xticks(bands.layout.xaxis.tickvals, [r"$\Gamma$", "$K$", "$M$", r"$\Gamma$"])
    band_axis.set(title=f"Bands at M = {sample['mass']:.3f}", ylabel=r"Energy $E/t_1$",
                  xlabel=f"Minimum direct gap: {sample['gap']:.6f} $t_1$")
    for shape in chern.layout.shapes:
        chern_axis.axvline(shape.x0, color="#8793a5", ls="--", lw=1)
    chern_axis.axvline(sample["mass"], color=INK, lw=1)
    chern_axis.set(title="Chern phase diagram", xlabel=r"Mass $M/t_1$", ylabel="Chern number",
                   yticks=[-1, 0, 1], ylim=(-1.3, 1.3))
    chern_axis.legend(loc="center", frameon=False)
    for axis in (band_axis, chern_axis):
        axis.grid(alpha=0.22)
    figure.savefig(Path(__file__).with_name("m3_snapshot.png"), dpi=180)
    plt.close(figure)


def main():
    path, distance, ticks = symmetry_path()
    valley_offsets = np.linspace(-0.28, 0.28, 81)
    valley_path = -K + np.column_stack([valley_offsets, np.zeros_like(valley_offsets)])
    samples = build_samples(path, valley_path)
    anchors = [next(i for i, sample in enumerate(samples) if sample["mass"] == mass)
               for mass in (-CRITICAL_MASS, 0.0, CRITICAL_MASS)]
    initial = samples[anchors[1]]
    bands = band_figure(distance, ticks, valley_offsets, initial)
    chern = chern_figure(samples, initial)
    band_plot = plot_embed(bands, "bands", include_plotlyjs=True)
    chern_plot = plot_embed(chern, "chern", include_plotlyjs=False)
    payload = json.dumps(dict(samples=samples, anchors=anchors), separators=(",", ":"), allow_nan=False)
    output = Path(__file__).with_name("m3_interactive.html")
    replacements = {
        "<!-- BAND_PLOT -->": band_plot,
        "<!-- CHERN_PLOT -->": chern_plot,
        "/* MODEL_DATA */": payload,
        "{MODEL_PARAMETERS}": f"t₁ = {T1:g}　·　t₂ = {T2:g}　·　φ = {PHI / np.pi:g}π",
        "{FHS_GRID}": f"{GRID_SIZE} × {GRID_SIZE}",
        "{GAP_GRID}": f"{GAP_GRID_SIZE} × {GAP_GRID_SIZE}",
        "{CRITICAL_VALUE}": f"{CRITICAL_MASS:.4f}",
        "{MASS_MIN}": f"{MASS_MIN:g}",
        "{MASS_MAX}": f"{MASS_MAX:+g}",
    }
    html = PAGE
    for placeholder, value in replacements.items():
        html = html.replace(placeholder, value)
    output.write_text(html, encoding="utf-8")
    snapshot = min((sample for sample in samples if sample["chern"][0] == -1),
                   key=lambda sample: sample["gap"])
    save_snapshot(bands, chern, snapshot)
    print(f"Wrote {output.name} with {len(samples)} slider positions and m3_snapshot.png.")


PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Haldane 模型的能带与陈数</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; padding: 28px 24px; background: #f1f4f9; color: #202b3b;
         font: 15px/1.5 'Segoe UI', 'Microsoft YaHei', sans-serif; }
  main { max-width: 1320px; margin: auto; }
  h1, h2, p { margin: 0; }
  h1 { font-size: clamp(24px, 4vw, 34px); letter-spacing: -.025em; }
  h2 { font-size: 19px; }
  .parameters, .caption { color: #56647a; }
  .parameters { margin-top: 4px; }
  .controls, .panel { background: #fff; border: 1px solid #d9e1ed; border-radius: 12px; }
  .controls { margin: 20px 0 15px; padding: 14px 18px; display: grid; gap: 16px;
              grid-template-columns: minmax(0, 1fr) auto; align-items: center; }
  .slider-label, .scale { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  output, .metric { font-variant-numeric: tabular-nums; }
  input[type=range] { width: 100%; display: block; margin: 8px 0 3px; accent-color: #2459a7; }
  .scale, .caption { font-size: 13px; }
  .jumps { display: flex; flex-wrap: wrap; gap: 7px; }
  button { background: #f7f9fd; color: #202b3b; border: 1px solid #cbd6e8;
           border-radius: 6px; padding: 7px 10px; font: inherit; cursor: pointer; }
  button:hover { background: #e8effa; }
  :focus-visible { outline: 2px solid #2459a7; outline-offset: 3px; }
  .panels { display: grid; gap: 15px; grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr); }
  .panel { min-width: 0; padding: 16px 12px 13px; }
  .panel h2, .panel .metric, .panel .caption { margin-inline: 12px; }
  .metric { margin-top: 4px; min-height: 23px; }
  .metric span { margin-right: 13px; }
  .lower { color: #2459a7; } .upper { color: #b94d26; }
  @media (max-width: 850px) { .panels { grid-template-columns: minmax(0, 1fr); } }
  @media (max-width: 550px) { body { padding: 16px 10px; } .controls { grid-template-columns: minmax(0, 1fr); } }
</style>
</head>
<body>
<main>
  <header>
    <h1>Haldane 模型的能带与陈数</h1>
    <p class="parameters">{MODEL_PARAMETERS}　·　FHS 网格 {FHS_GRID}</p>
  </header>
  <section class="controls" aria-label="质量项控制">
    <div>
      <div class="slider-label"><label for="mass">质量项 M/t₁</label><output id="mass-value" for="mass"></output></div>
      <input id="mass" type="range" min="0" step="1" disabled>
      <div class="scale"><span>{MASS_MIN}</span><span>{MASS_MAX}</span></div>
    </div>
    <div class="jumps">
      <button type="button" data-anchor="0" disabled>−{CRITICAL_VALUE} 闭隙</button>
      <button type="button" data-anchor="1" disabled>M = 0</button>
      <button type="button" data-anchor="2" disabled>+{CRITICAL_VALUE} 闭隙</button>
    </div>
  </section>
  <div class="panels">
    <section class="panel">
      <h2>能带与直接能隙</h2>
      <p class="metric">最小直接能隙 <strong id="gap"></strong> t₁</p>
      <!-- BAND_PLOT -->
      <p class="caption">主图沿用 M1 路径。旁图沿 kₓ 穿过 K′。能隙取 {GAP_GRID} 网格的最小值，包含两个谷。</p>
    </section>
    <section class="panel">
      <h2>陈数相图</h2>
      <p class="metric"><span class="lower">下带 <strong id="lower"></strong></span><span class="upper">上带 <strong id="upper"></strong></span></p>
      <!-- CHERN_PLOT -->
      <p class="caption">灰虚线标记解析闭隙 ±{CRITICAL_VALUE}。阶梯来自有限网格。闭隙时单带陈数未定义。</p>
    </section>
  </div>
</main>
<script>
  const {samples, anchors} = /* MODEL_DATA */;
  const slider = document.getElementById('mass');
  const massValue = document.getElementById('mass-value');
  const gap = document.getElementById('gap');
  const lower = document.getElementById('lower');
  const upper = document.getElementById('upper');
  const buttons = [...document.querySelectorAll('[data-anchor]')];

  function update() {
    const sample = samples[Number(slider.value)];
    const mass = sample.mass;
    massValue.value = mass.toFixed(6);
    slider.setAttribute('aria-valuetext', massValue.value);
    gap.textContent = sample.gap < 1e-10 ? '≈ 0' : sample.gap.toFixed(6);
    [lower.textContent, upper.textContent] = sample.chern.map(value => value ?? '未定义');
    Plotly.restyle('bands', {y: sample.bands}, [0, 1, 2, 3]);
    Plotly.restyle('chern', {x: [[mass, mass], [mass, mass]], y: [[-1.32, 1.18], sample.chern]}, [2, 3]);
  }

  slider.max = samples.length - 1;
  slider.value = anchors[1];
  let frame;
  slider.addEventListener('input', () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(update);
  });
  for (const button of buttons) {
    button.addEventListener('click', () => {
      slider.value = anchors[Number(button.dataset.anchor)];
      update();
    });
  }

  function enableWhenReady() {
    if (!document.getElementById('bands')._fullLayout || !document.getElementById('chern')._fullLayout) {
      requestAnimationFrame(enableWhenReady);
      return;
    }
    slider.disabled = false;
    for (const button of buttons) button.disabled = false;
    update();
  }
  requestAnimationFrame(enableWhenReady);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
