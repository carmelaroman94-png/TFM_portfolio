"""Plantilla HTML común de los dashboards: cabecera, navegación, tarjetas de gráficos y notas al pie."""
from __future__ import annotations

import html
import json

import plotly.graph_objects as go

PAGES = [
    ("index.html", "Inicio", ""),
    ("01_datos.html", "1 · Datos", "Análisis exploratorio"),
    ("02_clasificacion.html", "2 · Clasificación", "Análisis exploratorio"),
    ("03_inflacion.html", "3 · Inflación", "Análisis exploratorio"),
    ("04_extremos.html", "4 · Distribuciones y extremos", "Análisis exploratorio"),
    ("05_h1_perfiles.html", "5 · H1 Perfiles", "Modelado"),
    ("06_h2_h3_modelos.html", "6 · H2–H3 Modelos", "Modelado"),
]

CSS = """
:root{--forest:#26352F;--forest-2:#33463E;--cream:#F8F3ED;--surface:#fcfcfb;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;
--line:#e1e0d9;--accent:#2a78d6;--ok:#0ca30c;--warn:#eda100;--no:#898781;}
*{box-sizing:border-box}body{margin:0;background:var(--cream);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent)}
header{background:var(--forest);color:var(--cream);padding:28px 0 0}
.wrap{max-width:1180px;margin:0 auto;padding:0 28px}
.eyebrow{font-size:12px;letter-spacing:.12em;text-transform:uppercase;opacity:.75}
h1{font-size:30px;line-height:1.2;margin:6px 0 8px;font-weight:700}
.sub{max-width:820px;opacity:.85;margin:0 0 20px}
nav{display:flex;gap:4px;flex-wrap:wrap;margin-top:8px}
nav a{color:var(--cream);text-decoration:none;padding:10px 14px;border-radius:8px 8px 0 0;font-size:14px;opacity:.8}
nav a:hover{opacity:1;background:var(--forest-2)}nav a.on{background:var(--cream);color:var(--forest);opacity:1;font-weight:600}
nav .grp{align-self:center;font-size:11px;letter-spacing:.1em;text-transform:uppercase;opacity:.55;padding:0 4px 0 14px}
main{padding:28px 0 40px}
.objective{background:#fff;border-left:4px solid var(--forest);border-radius:8px;padding:18px 22px;margin-bottom:22px}
.objective h2{margin:0 0 6px;font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:var(--forest)}
.objective p{margin:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:22px}
.kpi{background:#fff;border-radius:10px;padding:14px 16px;border:1px solid var(--line)}
.kpi b{display:block;font-size:26px;line-height:1.1;font-weight:700;color:var(--forest)}
.kpi span{font-size:13px;color:var(--ink-2)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:20px 22px;margin-bottom:18px;min-width:0}
.card.full{grid-column:1/-1}
.card h3{margin:0 0 4px;font-size:18px}
.card .desc{margin:0 0 10px;color:var(--ink-2);font-size:14px}
.card .read{background:var(--cream);border-radius:8px;padding:10px 14px;margin-top:10px;font-size:14px}
.card .read b{color:var(--forest)}
.card .foot{margin:10px 0 0;font-size:12px;color:var(--muted)}
.chart{width:100%;min-height:380px}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:6px 0 4px;font-size:14px}
select,input[type=search]{font:inherit;padding:6px 10px;border:1px solid #c3c2b7;border-radius:6px;background:#fff}
.seg button{font:inherit;font-size:13px;padding:5px 12px;border:1px solid #c3c2b7;background:#fff;cursor:pointer}
.seg button:first-child{border-radius:6px 0 0 6px}.seg button:last-child{border-radius:0 6px 6px 0}
.seg button.on{background:var(--forest);color:#fff;border-color:var(--forest)}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line)}
th{color:var(--ink-2);font-weight:600;position:sticky;top:0;background:var(--surface)}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.tablebox{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:8px}
.cards3{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px;margin-bottom:22px}
.hyp{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;display:flex;flex-direction:column;gap:8px}
.hyp .tag{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:3px 10px;border-radius:999px;width:fit-content}
.tag.yes{background:#e3f4e3;color:#006300}.tag.partial{background:#fdf1d6;color:#7a5400}.tag.no{background:#efeee9;color:#52514e}
.hyp h4{margin:0;font-size:16px}.hyp p{margin:0;font-size:14px;color:var(--ink-2)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px}
.tile{display:block;background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px 20px;text-decoration:none;color:inherit;transition:border-color .15s}
.tile:hover{border-color:var(--forest)}.tile .num{font-size:12px;color:var(--muted);letter-spacing:.1em}
.tile h4{margin:4px 0 6px;font-size:17px;color:var(--forest)}.tile p{margin:0;font-size:14px;color:var(--ink-2)}
.btn{display:inline-block;background:var(--cream);color:var(--forest);padding:10px 18px;border-radius:8px;text-decoration:none;font-weight:600;margin:0 8px 8px 0}
.btn.ghost{background:transparent;color:var(--cream);border:1px solid rgba(248,243,237,.5)}
.pager{display:flex;justify-content:space-between;margin-top:10px;font-size:14px}
footer{border-top:1px solid var(--line);padding:22px 0 40px;color:var(--muted);font-size:13px}
@media(max-width:860px){.grid{grid-template-columns:1fr}h1{font-size:24px}}
"""

JS = """
const CFG={responsive:true,displaylogo:false,displayModeBar:false};
function draw(id,fig){try{const el=document.getElementById(id);const l=Object.assign({},fig.layout,{autosize:true});delete l.width;
 Plotly.react(el,fig.data,l,CFG).catch(e=>console.warn('chart '+id,e));}catch(e){console.warn('chart '+id+': '+e.message);}}
document.querySelectorAll('.chart[data-fig]').forEach(el=>draw(el.id,FIGS[el.dataset.fig]));
document.querySelectorAll('.seg').forEach(seg=>{seg.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>{
 seg.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 draw(seg.dataset.target,FIGS[b.dataset.fig]);}));});
"""


def fig_json(fig: go.Figure, height: int = 420) -> dict:
    fig.update_layout(height=height, width=None, margin=dict(l=60, r=24, t=30, b=56), title=None)
    return json.loads(fig.to_json())


def card(cid: str, title: str, desc: str, read: str = "", foot: str = "", full: bool = False,
         height: int = 420, controls: str = "", body: str = "", fig_key: str | None = None) -> str:
    chart = body or f'<div id="{cid}" class="chart" data-fig="{fig_key or cid}" style="height:{height}px"></div>'
    return (f'<section class="card{" full" if full else ""}"><h3>{title}</h3><p class="desc">{desc}</p>{controls}{chart}'
            + (f'<div class="read">{read}</div>' if read else "") + (f'<p class="foot">{foot}</p>' if foot else "") + "</section>")


def seg(target: str, options: list[tuple[str, str]]) -> str:
    buttons = "".join(f'<button data-fig="{k}" class="{"on" if i == 0 else ""}">{html.escape(lab)}</button>'
                      for i, (k, lab) in enumerate(options))
    return f'<div class="controls"><span class="seg" data-target="{target}">{buttons}</span></div>'


def kpis(items: list[tuple[str, str]]) -> str:
    return '<div class="kpis">' + "".join(f'<div class="kpi"><b>{v}</b><span>{l}</span></div>' for v, l in items) + "</div>"


def page(filename: str, title: str, subtitle: str, objective: str, body: str, figs: dict, repo_url: str,
         extra_js: str = "", sources: str = "") -> str:
    nav, group = "", ""
    for f, lab, grp in PAGES:
        if grp and grp != group:
            nav += f'<span class="grp">{grp}</span>'
            group = grp
        nav += f'<a href="{f}" class="{"on" if f == filename else ""}">{lab}</a>'
    idx = [f for f, _, _ in PAGES].index(filename)
    prev_link = f'<a href="{PAGES[idx - 1][0]}">← {PAGES[idx - 1][1]}</a>' if idx > 0 else "<span></span>"
    next_link = f'<a href="{PAGES[idx + 1][0]}">{PAGES[idx + 1][1]} →</a>' if idx < len(PAGES) - 1 else "<span></span>"
    obj = f'<div class="objective"><h2>Objetivo de este dashboard</h2><p>{objective}</p></div>' if objective else ""
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} · Multicausalidad de la inflación</title><meta name="theme-color" content="#26352F">
<style>{CSS}</style><script src="assets/plotly.min.js"></script></head><body>
<header><div class="wrap"><div class="eyebrow">Trabajo Final de Máster · Ciencia de Datos</div><h1>{title}</h1><p class="sub">{subtitle}</p>
<nav>{nav}</nav></div></header>
<main><div class="wrap">{obj}{body}<div class="pager">{prev_link}{next_link}</div></div></main>
<footer><div class="wrap">{sources}<br>Código, datos procesados y notebooks: <a href="{repo_url}">{repo_url}</a> · Elaboración propia.</div></footer>
<script>const FIGS={json.dumps(figs, separators=(",", ":"))};</script><script>{JS}{extra_js}</script></body></html>"""
