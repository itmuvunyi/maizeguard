"""MaizeGuard dashboard (Streamlit).   Run:  streamlit run ui/app.py

Works with Streamlit >= 1.36. Charts use Altair (installed with Streamlit).
"""
import html
import os
from urllib.parse import quote

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
CLASSES = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Per-class colour + short field description, used everywhere so a class
# always looks the same (charts, result card, sidebar guide).
CLASS_META = {
    "Blight": {"name": "Leaf Blight", "color": "#C9782F",
               "desc": "Long, cigar-shaped grey-green to tan lesions."},
    "Common_Rust": {"name": "Common Rust", "color": "#B23A26",
                    "desc": "Small powdery cinnamon-brown pustules on both leaf sides."},
    "Gray_Leaf_Spot": {"name": "Gray Leaf Spot", "color": "#66758A",
                       "desc": "Narrow rectangular grey lesions bounded by the veins."},
    "Healthy": {"name": "Healthy", "color": "#2F8F55",
                "desc": "Even green colour, no lesions or pustules."},
}

st.set_page_config(page_title="MaizeGuard · Maize disease detection", page_icon=":material/eco:",
                   layout="wide", initial_sidebar_state="expanded")

# ======================================================================= theme
# Two palettes. The sidebar toggle stores the choice in st.session_state["dark_mode"].
THEMES = {
    "light": dict(bg="#FAFBF9", surface="#FFFFFF", surface2="#F3F7F2", sidebar="#F4F7F3", border="#E1E8DF",
                  ink="#14261A", body="#23382A", muted="#5E6E63", grid="#E4EAE2", brand="#2F8F55",
                  brand_soft="#E6F2E9", brand_ink="#17603A", input="#F0F4EF", dashed="#D3DDD1",
                  shadow="rgba(20,38,26,.04)", heat_lo="#F2F7F2", heat_hi="#145C33",
                  ok="#1E8449", warn="#B86E00", bad="#C0392B"),
    "dark": dict(bg="#0D1410", surface="#141E18", surface2="#1A2720", sidebar="#101913", border="#263529",
                 ink="#E7F0E9", body="#C8D6CC", muted="#8DA294", grid="#223026", brand="#3FAF6C",
                 brand_soft="#1C3526", brand_ink="#86DDA8", input="#1A2720", dashed="#2F4236",
                 shadow="rgba(0,0,0,.35)", heat_lo="#1A2720", heat_hi="#3FAF6C",
                 ok="#5DD38C", warn="#F2B544", bad="#FF7A6B"),
}
DARK = bool(st.session_state.get("dark_mode", False))
T = THEMES["dark" if DARK else "light"]
INK, MUTED, GRID, BRAND = T["ink"], T["muted"], T["grid"], T["brand"]

# ======================================================================= style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input { font-family: 'Inter', system-ui, -apple-system, sans-serif; }
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ---------- hero ---------- */
.mg-hero { background: linear-gradient(120deg, #0F3520 0%, #1C5A35 55%, #2F8F55 100%);
  border-radius: 20px; padding: 28px 32px; color: #fff; display: flex; gap: 24px;
  align-items: center; justify-content: space-between; flex-wrap: wrap;
  box-shadow: 0 10px 30px -12px rgba(15,53,32,.45); margin-bottom: 1.2rem; }
.mg-hero h1 { color: #fff !important; font-size: 1.9rem; font-weight: 800; margin: 0;
  letter-spacing: -0.02em; padding: 0; }
.mg-hero p { color: rgba(255,255,255,.82) !important; margin: 6px 0 0; font-size: .98rem; }
.mg-hero .chips { display: flex; gap: 8px; flex-wrap: wrap; }
.mg-chip { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22);
  border-radius: 999px; padding: 6px 14px; font-size: .82rem; font-weight: 600; color: #fff;
  display: inline-flex; align-items: center; gap: 8px; white-space: nowrap; }
.mg-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 4px; background: var(--mg-surface); padding: 6px;
  border-radius: 14px; border: 1px solid var(--mg-border); }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 20px; border-radius: 10px;
  font-weight: 600; color: var(--mg-muted); background: transparent; }
.stTabs [data-baseweb="tab"] p { font-weight: 600; font-size: .93rem; }
.stTabs [aria-selected="true"] { background: var(--mg-brand-soft) !important; color: var(--mg-brand-ink) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.3rem; }

/* ---------- metric cards ---------- */
[data-testid="stMetric"] { background: var(--mg-surface); border: 1px solid var(--mg-border); border-radius: 14px;
  padding: 16px 18px; box-shadow: 0 1px 2px var(--mg-shadow); height: 100%; }
[data-testid="stMetricLabel"] p { color: var(--mg-muted); font-size: .8rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .04em; }
[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 700; color: var(--mg-ink); }

/* ---------- sections, cards, callouts ---------- */
.mg-eyebrow { color: var(--mg-brand); font-size: .74rem; font-weight: 700; letter-spacing: .1em;
  text-transform: uppercase; margin-bottom: 2px; }
.mg-h { font-size: 1.22rem; font-weight: 700; color: var(--mg-ink); margin: 0; letter-spacing: -0.01em; }
.mg-sub { color: var(--mg-muted); font-size: .9rem; margin: 2px 0 12px; }
.mg-section { margin: 1.4rem 0 .2rem; }
.mg-story { background: var(--mg-surface2); border-left: 4px solid var(--mg-brand); border-radius: 10px;
  padding: 14px 16px; color: var(--mg-body); font-size: .92rem; line-height: 1.55; }
.mg-story b { color: var(--mg-ink); }
.mg-story .tag { font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: var(--mg-brand);
  text-transform: uppercase; display: block; margin-bottom: 6px; }
.mg-empty { border: 2px dashed var(--mg-dashed); border-radius: 16px; padding: 48px 24px;
  text-align: center; color: var(--mg-muted); background: var(--mg-surface2); }
.mg-empty .big { font-size: 2.2rem; }
.mg-empty b { color: var(--mg-ink); display: block; margin: 6px 0 4px; font-size: 1rem; }

/* ---------- prediction result ---------- */
.mg-result { border-radius: 16px; padding: 20px 22px; color: #fff; margin-bottom: 14px; }
.mg-result .k { font-size: .74rem; letter-spacing: .1em; text-transform: uppercase;
  font-weight: 700; opacity: .85; }
.mg-result .v { font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin: 2px 0 10px; }
.mg-bar { height: 8px; border-radius: 99px; background: rgba(255,255,255,.28); overflow: hidden; }
.mg-bar > div { height: 100%; background: #fff; border-radius: 99px; }
.mg-result .conf { display: flex; justify-content: space-between; font-size: .86rem;
  margin-top: 6px; font-weight: 600; }
.mg-advice { background: var(--mg-surface); border: 1px solid var(--mg-border); border-radius: 14px;
  padding: 14px 16px; font-size: .93rem; line-height: 1.55; color: var(--mg-body); margin-bottom: 8px; }
.mg-advice .t { font-weight: 700; color: var(--mg-ink); margin-bottom: 4px; }

/* ---------- stepper ---------- */
.mg-steps { display: flex; gap: 10px; margin-bottom: 1rem; flex-wrap: wrap; }
.mg-step { flex: 1; min-width: 180px; background: var(--mg-surface); border: 1px solid var(--mg-border);
  border-radius: 14px; padding: 12px 14px; display: flex; gap: 12px; align-items: center; }
.mg-step .n { width: 30px; height: 30px; border-radius: 50%; background: var(--mg-brand-soft); color: var(--mg-brand-ink);
  font-weight: 800; display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
.mg-step .s { font-weight: 700; color: var(--mg-ink); font-size: .9rem; }
.mg-step .d { color: var(--mg-muted); font-size: .78rem; }

/* ---------- icons ---------- */
.mg-ico { vertical-align: -0.18em; flex-shrink: 0; }
.mg-logo { display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px;
  border-radius: 9px; background: #2F8F55; color: #fff; margin-right: 10px; vertical-align: middle; }
.mg-logo.lg { width: 44px; height: 44px; border-radius: 12px; background: rgba(255,255,255,.16);
  border: 1px solid rgba(255,255,255,.25); margin-right: 14px; }
.mg-advice .t { display: flex; align-items: center; gap: 8px; }
.mg-advice .t .mg-ico { color: var(--mg-warn); }
.mg-empty .big { color: var(--mg-brand); display: inline-flex; width: 60px; height: 60px; border-radius: 16px;
  background: var(--mg-brand-soft); align-items: center; justify-content: center; }
.mg-hero h1 { display: flex; align-items: center; }

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] { background: var(--mg-sidebar); border-right: 1px solid var(--mg-border); }
.mg-side-brand { display: flex; align-items: center; font-weight: 800; font-size: 1.25rem; color: var(--mg-ink); letter-spacing: -0.02em; }
.mg-pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
  border-radius: 999px; font-weight: 600; font-size: .82rem; }
.mg-legend { display: flex; gap: 10px; align-items: flex-start; margin: 10px 0; }
.mg-legend .sw { width: 12px; height: 12px; border-radius: 4px; margin-top: 4px; flex-shrink: 0; }
.mg-legend .nm { font-weight: 600; color: var(--mg-ink); font-size: .88rem; }
.mg-legend .ds { color: var(--mg-muted); font-size: .8rem; line-height: 1.4; }

/* ---------- tables ---------- */
.mg-scroll { overflow-x: auto; }
.mg-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .86rem;
  background: var(--mg-surface); border: 1px solid var(--mg-border); border-radius: 12px; overflow: hidden; }
.mg-table th { text-align: left; font-weight: 600; color: var(--mg-muted); font-size: .72rem; white-space: nowrap;
  text-transform: uppercase; letter-spacing: .05em; padding: 10px 12px; background: var(--mg-surface2);
  border-bottom: 1px solid var(--mg-border); }
.mg-table td { padding: 10px 12px; color: var(--mg-ink); border-bottom: 1px solid var(--mg-border);
  font-variant-numeric: tabular-nums; }
.mg-table th, .mg-table td { border-top: none !important; border-left: none !important; border-right: none !important; }
.mg-table tr:last-child td { border-bottom: none; }
.mg-table td:first-child { white-space: nowrap; }
.mg-table.compact th, .mg-table.compact td { padding: 9px 8px; }
.mg-mini { display: flex; align-items: center; gap: 6px; }
.mg-mini .track { flex: none; width: 40px; height: 6px; background: var(--mg-surface2); border-radius: 99px; overflow: hidden; }
.mg-mini .fill { height: 100%; border-radius: 99px; background: var(--mg-brand); }
.mg-mini span { width: 36px; text-align: right; }
.mg-badge { display: inline-flex; align-items: center; gap: 6px; padding: 3px 10px; border-radius: 99px;
  font-size: .76rem; font-weight: 600; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)


# Theme variables for the custom CSS above, plus Streamlit's own widgets.
_vars = ";".join(f"--mg-{k.replace('_', '-')}:{v}" for k, v in T.items())
st.markdown(f"""<style>
:root, .stApp {{ {_vars}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background: var(--mg-bg) !important; color: var(--mg-ink); }}
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li,
.stApp [data-testid="stWidgetLabel"] p, .stApp label, .stApp h1, .stApp h2, .stApp h3 {{ color: var(--mg-ink); }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--mg-muted) !important; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ border-color: var(--mg-border) !important; }}
hr {{ border-color: var(--mg-border) !important; background: var(--mg-border) !important; }}
[data-baseweb="select"] > div, [data-testid="stFileUploaderDropzone"], .stCodeBlock pre, .stCodeBlock code,
[data-testid="stCode"] pre {{ background: var(--mg-input) !important; color: var(--mg-ink) !important;
  border-color: var(--mg-border) !important; }}
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] svg, [data-baseweb="select"] svg {{ color: var(--mg-muted) !important; }}
[data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFile"] div {{ color: var(--mg-ink) !important; }}
[data-testid="stFileUploaderFile"] svg, [data-testid="stFileUploaderDeleteBtn"] button {{ color: var(--mg-muted) !important; }}
button[kind="secondary"], [data-testid="baseButton-secondary"] {{ background: var(--mg-surface) !important;
  color: var(--mg-ink) !important; border: 1px solid var(--mg-border) !important; }}
button[kind="secondary"]:hover, [data-testid="baseButton-secondary"]:hover {{ border-color: var(--mg-brand) !important;
  color: var(--mg-brand) !important; }}
button[kind="secondary"]:disabled {{ opacity: .55; }}
[data-baseweb="popover"] ul, [data-baseweb="popover"] li, [data-baseweb="menu"] {{
  background: var(--mg-surface) !important; color: var(--mg-ink) !important; }}
[data-baseweb="popover"] li:hover {{ background: var(--mg-surface2) !important; }}
[data-testid="stToast"] {{ background: var(--mg-surface) !important; color: var(--mg-ink) !important;
  border: 1px solid var(--mg-border); }}
[data-testid="stAlert"] p {{ color: #000000 !important; }}
[data-testid="stAlert"] svg {{ color: #000000 !important; }}
[data-testid="stHeader"] button, [data-testid="stSidebarCollapseButton"] button,
[data-testid="collapsedControl"] button {{ color: var(--mg-muted) !important; }}
.stTooltipIcon svg {{ color: var(--mg-muted); }}
[data-baseweb="progress-bar"] > div > div {{ background: var(--mg-border) !important; }}
[data-testid="stFileUploaderFile"] small {{ color: var(--mg-muted) !important; }}
</style>""", unsafe_allow_html=True)


# ===================================================================== helpers

# Outline icons (Lucide style) for the custom HTML parts of the page.
ICONS = {
    "leaf": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "image": '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="9" cy="9" r="2"/><path d="m21 15-3.1-3.1a2 2 0 0 0-2.8 0L6 21"/>',
    "arrow_left": '<path d="m12 19-7-7 7-7"/><path d="M19 12H5"/>',
    "bulb": '<path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/>',
    "folder": '<path d="M20 20a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.69-.9L9.6 3.9A2 2 0 0 0 7.93 3H4a2 2 0 0 0-2 2v13a2 2 0 0 0 2 2Z"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5"/><path d="M12 3v12"/>',
    "sliders": '<path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 1v3M15 1v3M9 20v3M15 20v3M20 9h3M20 14h3M1 9h3M1 14h3"/>',
    "shield": '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/><path d="m9 12 2 2 4-4"/>',
    "server": '<rect x="2" y="2" width="20" height="8" rx="2"/><rect x="2" y="14" width="20" height="8" rx="2"/><path d="M6 6h.01M6 18h.01"/>',
    "refresh": '<path d="M21 12a9 9 0 0 0-9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/><path d="M3 12a9 9 0 0 0 9 9 9.75 9.75 0 0 0 6.74-2.74L21 16"/><path d="M16 16h5v5"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "scan": '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/>',
    "chart": '<path d="M3 3v16a2 2 0 0 0 2 2h16"/><path d="M18 17V9M13 17V5M8 17v-3"/>',
    "retrain": '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/><path d="M8 16H3v5"/>',
    "check": '<path d="M20 6 9 17l-5-5"/>',
    "timer": '<path d="M10 2h4M12 14l3-3"/><circle cx="12" cy="14" r="8"/>',
    "tag": '<path d="M12.6 2.6A2 2 0 0 0 11.2 2H4a2 2 0 0 0-2 2v7.2a2 2 0 0 0 .6 1.4l8.7 8.7a2.4 2.4 0 0 0 3.4 0l6.6-6.6a2.4 2.4 0 0 0 0-3.4z"/><circle cx="7.5" cy="7.5" r=".5" fill="currentColor"/>',
}


def svg(name, size=20):
    return (f'<svg class="mg-ico" xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
            f'viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')

def api(method, path, **kw):
    """Call the API. Returns (json, error_message)."""
    try:
        r = requests.request(method, API + path, timeout=kw.pop("timeout", 60), **kw)
        if r.ok:
            return r.json(), None
        try:
            return None, r.json().get("detail", r.text)
        except ValueError:
            return None, r.text
    except requests.RequestException as exc:
        return None, f"API not reachable: {exc}"


def fmt_uptime(sec):
    d, rem = divmod(int(sec), 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}d {h}h {m}m" if d else f"{h}h {m}m {s}s"


def nice(cls):
    return CLASS_META.get(cls, {}).get("name", str(cls).replace("_", " "))


def color_of(cls):
    return CLASS_META.get(cls, {}).get("color", "#7A8A80")


def _mask(name):
    """CSS mask-image so a CSS-drawn icon takes the text colour of the button/tab."""
    raw = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="black" '
           f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg>')
    url = "url(\"data:image/svg+xml;utf8," + quote(raw) + "\")"
    return (f'content:"";display:inline-block;width:1.05em;height:1.05em;margin-right:.5em;'
            f'vertical-align:-0.16em;background-color:currentColor;'
            f'-webkit-mask:{url} center/contain no-repeat;mask:{url} center/contain no-repeat;')


TAB_ICONS = ["activity", "scan", "chart", "retrain"]
BUTTON_ICONS = ["refresh", "search", "upload", "retrain"]
_EC = [".element-container", '[data-testid="stElementContainer"]']  # 1.36 and newer class names
_icon_css = "".join(
    f'.stTabs [data-baseweb="tab"]:nth-of-type({i + 1}) p::before{{{_mask(n)}}}' for i, n in enumerate(TAB_ICONS))
for n in BUTTON_ICONS:
    for ec in _EC:
        _icon_css += (f"{ec}:has(> div > .mg-ib-{n}), {ec}:has(.mg-ib-{n}){{display:none}}"
                      f"{ec}:has(.mg-ib-{n}) + {ec} button p::before{{{_mask(n)}}}")
st.markdown(f"<style>{_icon_css}</style>", unsafe_allow_html=True)


def icon_button(icon, label, **kw):
    """st.button with an icon (Streamlit 1.36 has no icon= argument on buttons)."""
    st.markdown(f'<span class="mg-ib-{icon}"></span>', unsafe_allow_html=True)
    return st.button(label, **kw)


def html_block(s):
    """Render raw HTML. Lines are stripped so Markdown never turns them into code blocks."""
    st.markdown("".join(line.strip() for line in s.splitlines()), unsafe_allow_html=True)


def esc(v):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else html.escape(str(v))



def html_table(rows, headers, compact=False):
    """rows: list of lists of ready-made HTML cell strings."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    table_class = "mg-table compact" if compact else "mg-table"
    html_block(f'<div class="mg-scroll"><table class="{table_class}"><thead><tr>{head}</tr></thead>'
               f'<tbody>{body}</tbody></table></div>')



def mini_bar(pct, color=None):
    return (f'<div class="mg-mini"><div class="track"><div class="fill" style="width:{max(0, min(pct, 100)):.0f}%;'
            f'{f"background:{color};" if color else ""}"></div></div><span>{pct:.0f}%</span></div>')


def badge(text, color):
    return f'<span class="mg-badge" style="color:{color};background:{color}22">{esc(text)}</span>'


def section(title, sub=None, eyebrow=None):
    html_block(f"""<div class="mg-section">
        {f'<div class="mg-eyebrow">{eyebrow}</div>' if eyebrow else ''}
        <div class="mg-h">{title}</div>
        {f'<div class="mg-sub">{sub}</div>' if sub else '<div style="height:10px"></div>'}
    </div>""")


def story(text, tag="What this tells us"):
    html_block(f'<div class="mg-story"><span class="tag">{tag}</span>{text}</div>')


def empty_state(icon, title, text):
    html_block(f'<div class="mg-empty"><div class="big">{svg(icon, 30)}</div><b>{title}</b>{text}</div>')


def show_image(file, caption=None):
    try:  # Streamlit >= 1.40
        st.image(file, caption=caption, use_container_width=True)
    except TypeError:  # Streamlit 1.36 - 1.39
        st.image(file, caption=caption, use_column_width=True)


# ---------------------------------------------------------------------- charts
CLASS_SCALE = alt.Scale(domain=[nice(c) for c in CLASSES], range=[color_of(c) for c in CLASSES])


def styled(chart, height=280):
    return (chart.properties(height=height, background="transparent")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, labelFont="Inter", titleFont="Inter",
                            labelFontSize=12, titleFontSize=12, titleFontWeight=500,
                            gridColor=GRID, domainColor=GRID, tickColor=GRID)
            .configure_legend(labelColor=INK, titleColor=MUTED, labelFont="Inter", titleFont="Inter",
                              orient="top", title=None, labelFontSize=12, symbolType="circle")
            .configure_text(font="Inter"))


def chart(c):
    st.altair_chart(c, use_container_width=True, theme=None)


def class_order(keys):
    keys = list(keys)
    return [nice(c) for c in CLASSES if c in keys] + [nice(c) for c in keys if c not in CLASSES]


def class_bars(series, y_title, fmt=",.0f", height=280, wrap=False):
    df = pd.DataFrame({"name": [nice(k) for k in series.index], "value": series.values.astype(float)})
    base = alt.Chart(df).encode(
        x=alt.X("name:N", sort=class_order(series.index), title=None,
                axis=alt.Axis(labelAngle=0, labelPadding=8, ticks=False,
                              labelExpr="split(datum.label, ' ')" if wrap else "datum.label"),
                scale=alt.Scale(paddingInner=0.4)),
        y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt, tickCount=5)),
        tooltip=[alt.Tooltip("name:N", title="Class"), alt.Tooltip("value:Q", title=y_title, format=fmt)])
    bars = base.mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7).encode(
        color=alt.Color("name:N", scale=CLASS_SCALE, legend=None))
    labels = base.mark_text(dy=-9, fontSize=12, fontWeight=600, color=INK).encode(
        text=alt.Text("value:Q", format=fmt))
    chart(styled(bars + labels, height))


def grouped_bars(frame, colors, y_title, fmt=",.1f", height=280):
    long = frame.rename_axis("cls").reset_index().melt(id_vars="cls", var_name="series", value_name="value")
    long["name"] = long["cls"].map(nice)
    c = alt.Chart(long).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
        x=alt.X("name:N", sort=class_order(frame.index), title=None,
                axis=alt.Axis(labelAngle=0, labelPadding=8, ticks=False), scale=alt.Scale(paddingInner=0.3)),
        xOffset=alt.XOffset("series:N", sort=list(colors)),
        y=alt.Y("value:Q", title=y_title, axis=alt.Axis(format=fmt, tickCount=5)),
        color=alt.Color("series:N", sort=list(colors),
                        scale=alt.Scale(domain=list(colors), range=list(colors.values()))),
        tooltip=[alt.Tooltip("name:N", title="Class"), alt.Tooltip("series:N", title="Feature"),
                 alt.Tooltip("value:Q", title="Value", format=fmt)])
    chart(styled(c, height))


def probability_bars(probs):
    df = pd.DataFrame({"name": [nice(k) for k in probs], "p": [float(v) for v in probs.values()]})
    df = df.sort_values("p", ascending=False)
    base = alt.Chart(df).encode(
        y=alt.Y("name:N", sort="-x", title=None, axis=alt.Axis(ticks=False, domain=False, labelPadding=8)),
        x=alt.X("p:Q", title=None, scale=alt.Scale(domain=[0, 1.12]),
                axis=alt.Axis(format="%", tickCount=5, labels=False, grid=False, domain=False, ticks=False)),
        tooltip=[alt.Tooltip("name:N", title="Class"), alt.Tooltip("p:Q", title="Probability", format=".1%")])
    bars = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, height=22).encode(
        color=alt.Color("name:N", scale=CLASS_SCALE, legend=None))
    labels = base.mark_text(align="left", dx=6, fontSize=12, fontWeight=600, color=INK).encode(
        text=alt.Text("p:Q", format=".1%"))
    chart(styled(bars + labels, height=44 * len(df)))


def confusion_heatmap(cm, names):
    cm = np.asarray(cm, dtype=float)
    rows = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, rows, out=np.zeros_like(cm), where=rows > 0)
    order = [nice(n) for n in names]
    df = pd.DataFrame([{"true": nice(t), "pred": nice(p), "count": int(cm[i, j]), "share": share[i, j]}
                       for i, t in enumerate(names) for j, p in enumerate(names)])
    base = alt.Chart(df).encode(
        x=alt.X("pred:N", sort=order, title="Predicted class",
                axis=alt.Axis(labelAngle=0, ticks=False, domain=False, labelPadding=6,
                              labelExpr="split(datum.label, ' ')", labelLineHeight=13)),
        y=alt.Y("true:N", sort=order, title="True class", axis=alt.Axis(ticks=False, domain=False)))
    rect = base.mark_rect(cornerRadius=6, stroke=T["surface"], strokeWidth=3).encode(
        color=alt.Color("share:Q", legend=None, scale=alt.Scale(domain=[0, 1], range=[T["heat_lo"], T["heat_hi"]])),
        tooltip=[alt.Tooltip("true:N", title="True"), alt.Tooltip("pred:N", title="Predicted"),
                 alt.Tooltip("count:Q", title="Images"), alt.Tooltip("share:Q", title="Share of row", format=".1%")])
    text = base.mark_text(fontSize=14, fontWeight=700).encode(
        text="count:Q",
        color=alt.condition("datum.share > 0.5", alt.value("#FFFFFF" if not DARK else "#0D1410"), alt.value(INK)))
    chart(styled(rect + text, height=330))


def training_curve(h, cols, title):
    cols = [c for c in cols if c in h]
    if not cols:
        return
    d = h[cols].copy()
    d["epoch"] = np.arange(1, len(d) + 1)
    long = d.melt(id_vars="epoch", var_name="key", value_name="value")
    long["split"] = long["key"].map(lambda k: "Validation" if k.startswith("val_") else "Training")
    c = alt.Chart(long).mark_line(strokeWidth=2.6, point=alt.OverlayMarkDef(size=36, filled=True)).encode(
        x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1, format="d")),
        y=alt.Y("value:Q", title=title, scale=alt.Scale(zero=False)),
        color=alt.Color("split:N", scale=alt.Scale(domain=["Training", "Validation"],
                                                   range=[BRAND, "#E0A100"])),
        tooltip=["epoch:Q", "split:N", alt.Tooltip("value:Q", format=".4f")])
    chart(styled(c, 260))


# ================================================================ shared data
health, health_err = api("GET", "/health", timeout=10)
stats, _ = api("GET", "/stats", timeout=10)
metrics, _ = api("GET", "/metrics", timeout=10)

online = health is not None and health.get("status") == "ok"
if health_err:
    state_txt, state_col = "Offline", T["bad"]
elif online:
    state_txt, state_col = "Online", T["ok"]
else:
    state_txt, state_col = "No model loaded", T["warn"]
state_bg = state_col + "22"  # same colour, ~13% opacity

# ===================================================================== sidebar
with st.sidebar:
    html_block(f'<div class="mg-side-brand"><span class="mg-logo">{svg("leaf", 18)}</span>MaizeGuard</div>'
               '<div style="color:var(--mg-muted);font-size:.85rem;margin-bottom:14px">ML pipeline control centre</div>')
    html_block(f'<span class="mg-pill" style="background:{state_bg};color:{state_col}">'
               f'<span class="mg-dot" style="background:{state_col}"></span>Model {state_txt}</span>')
    st.write("")
    st.toggle("Dark mode", key="dark_mode")
    if health:
        st.caption(f"Version **{health.get('model_version') or 'none'}** · "
                   f"up {fmt_uptime(health.get('uptime_seconds', 0))}")
    st.caption("API endpoint")
    st.code(API, language=None)
    if icon_button("refresh", "Refresh data", use_container_width=True):
        st.rerun()

    st.divider()
    st.markdown("**Disease guide**")
    for c in CLASSES:
        m = CLASS_META[c]
        html_block(f'<div class="mg-legend"><span class="sw" style="background:{m["color"]}"></span>'
                   f'<div><div class="nm">{m["name"]}</div><div class="ds">{m["desc"]}</div></div></div>')
    st.divider()
    st.caption("Built for Rwandan maize farmers · TensorFlow · FastAPI · Docker")

# ======================================================================== hero
version = (health or {}).get("model_version") or "—"
container = (health or {}).get("container") or "—"
html_block(f"""
<div class="mg-hero">
  <div>
    <h1><span class="mg-logo lg">{svg("leaf", 26)}</span>MaizeGuard</h1>
    <p>Detect maize leaf diseases from a single photo, and keep the model learning from new field data.</p>
  </div>
  <div class="chips">
    <span class="mg-chip"><span class="mg-dot" style="background:{'#7CF0A6' if online else '#FFB4A8'}"></span>{state_txt}</span>
    <span class="mg-chip">{svg("tag", 14)}Model {html.escape(str(version))}</span>
    <span class="mg-chip">{svg("server", 14)}Container {html.escape(str(container))}</span>
  </div>
</div>""")

tab_status, tab_predict, tab_viz, tab_retrain = st.tabs(
    ["Monitoring", "Predict", "Data insights", "Upload & retrain"])

# ================================================================== monitoring
with tab_status:
    if health_err:
        st.error(f"**The model API is down.** {health_err}", icon=":material/cloud_off:")
        st.caption(f"Check that the API is running at `{API}` and press **Refresh data** in the sidebar.")
    else:
        section("Service health", "Live status of the prediction service.", "Uptime")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model status", "Online" if online else "No model file")
        c2.metric("Up-time", fmt_uptime(health["uptime_seconds"]))
        c3.metric("Model version", health["model_version"] or "none")
        c4.metric("Answered by container", health["container"])
        st.caption(f"Running since {health['started_at']}. With several containers behind the load "
                   "balancer, refresh to see different containers answer.")

    if stats:
        p = stats["predictions"]
        section("Production traffic", "What the deployed model has been doing.", "Usage")
        c1, c2, c3 = st.columns(3)
        c1.metric("Predictions served", f"{p['total']:,}")
        c2.metric("Avg model latency", f"{p['avg_latency_ms']:.0f} ms" if p["avg_latency_ms"] else "n/a")
        c3.metric("Avg confidence", f"{p['avg_confidence']:.1%}" if p["avg_confidence"] else "n/a")
        if p["by_class"]:
            with st.container(border=True):
                st.markdown("**Prediction mix in production**")
                class_bars(pd.Series({k: v["count"] for k, v in p["by_class"].items()}), "Predictions")

    if metrics and metrics.get("test"):
        t = metrics["test"]
        section("Model evaluation in production",
                f"Held-out test set · {t['n_images']} images · trained {metrics['trained_at']} · {metrics['note']}",
                "Evaluation")
        cols = st.columns(5)
        for col, key in zip(cols, ["accuracy", "precision", "recall", "f1", "loss"]):
            label = "F1 score" if key == "f1" else key.capitalize()
            col.metric(label, f"{t[key]:.3f}")

        st.write("")
        left, right = st.columns([1, 1.05], gap="large")
        with left:
            with st.container(border=True):
                st.markdown("**Confusion matrix**")
                st.caption("Rows are the true class, columns the predicted class. A strong diagonal is good.")
                confusion_heatmap(t["confusion_matrix"], t["class_names"])
        with right:
            with st.container(border=True):
                st.markdown("**Per-class performance**")
                st.caption("Computed from the confusion matrix.")
                cm = np.asarray(t["confusion_matrix"], dtype=float)
                tp = np.diag(cm)
                rec = np.divide(tp, cm.sum(1), out=np.zeros_like(tp), where=cm.sum(1) > 0)
                prec = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
                f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
                per = pd.DataFrame({"Class": [nice(n) for n in t["class_names"]],
                                    "Precision": prec * 100, "Recall": rec * 100, "F1": f1 * 100,
                                    "Images": cm.sum(1).astype(int)})
                html_table([[f'<span class="mg-dot" style="background:{color_of(n)};margin-right:8px"></span>'
                             f'{esc(nice(n))}', mini_bar(pr * 100), mini_bar(rc * 100), mini_bar(f * 100)]
                            for n, pr, rc, f in zip(t["class_names"], prec, rec, f1)],
                           ["Class", "Precision", "Recall", "F1"], compact=True)
                st.write("")
                weakest = per.loc[per["Recall"].idxmin()]
                story(f"The model is weakest on <b>{weakest['Class']}</b> "
                      f"(recall {weakest['Recall']:.0f}%). New uploads of this class are the most "
                      "valuable for the next retraining run.", tag="Where to improve")
    elif not health_err:
        st.info("No evaluation metrics yet. Train or retrain a model to populate this section.")

# ===================================================================== predict
with tab_predict:
    section("Diagnose a maize leaf", "Upload a clear photo of a single leaf. The model returns the most "
            "likely disease, its confidence and what to do next.", "Prediction")
    left, right = st.columns([1, 1.1], gap="large")

    with left:
        with st.container(border=True):
            img = st.file_uploader("Leaf photo (JPG or PNG)", type=["jpg", "jpeg", "png"], key="predict_file")
            if img:
                show_image(img)
                st.caption(f"{img.name} · {img.size / 1024:.0f} KB")
            run = icon_button("search", "Analyse leaf", type="primary", use_container_width=True,
                            disabled=img is None)

    if run and img:
        with st.spinner("Analysing the leaf..."):
            res, err = api("POST", "/predict", files={"file": (img.name, img.getvalue(), img.type)})
        st.session_state["prediction"] = {"key": (img.name, img.size), "res": res, "err": err}

    with right:
        pred = st.session_state.get("prediction")
        if not img:
            empty_state("image", "No photo yet", "Upload a leaf photo on the left to get a diagnosis.")
        elif not pred or pred["key"] != (img.name, img.size):
            empty_state("arrow_left", "Ready to analyse", "Press <b style='display:inline'>Analyse leaf</b> to run the model.")
        elif pred["err"]:
            st.error(f"**Prediction failed.** {pred['err']}", icon=":material/error:")
        else:
            res = pred["res"]
            lbl, conf = res["label"], float(res["confidence"])
            healthy = lbl == "Healthy"
            col = color_of(lbl)
            verdict = "No disease detected" if healthy else "Disease detected"
            html_block(f"""
            <div class="mg-result" style="background:linear-gradient(120deg,{col} 0%,{col}CC 100%)">
              <div class="k">{verdict}</div>
              <div class="v">{html.escape(nice(lbl))}</div>
              <div class="mg-bar"><div style="width:{conf * 100:.1f}%"></div></div>
              <div class="conf"><span>Model confidence</span><span>{conf:.1%}</span></div>
            </div>""")
            if conf < 0.6:
                st.warning("Low confidence. Try a sharper photo of one leaf in good daylight.", icon=":material/warning:")
            html_block(f'<div class="mg-advice"><div class="t">{svg("bulb", 16)}Recommended action</div>'
                       f'{html.escape(str(res["advice"]))}</div>')
            with st.container(border=True):
                st.markdown("**Probability per class**")
                probability_bars(res["probabilities"])
            html_block(f'<div style="color:{MUTED};font-size:.85rem;display:flex;align-items:center;gap:6px">'
                       f'{svg("timer", 15)}Model time {res["latency_ms"]} ms · served by container '
                       f'{html.escape(str(res["container"]))}</div>')

# ==================================================================== insights
with tab_viz:
    ins, err = api("GET", "/insights", timeout=10)
    if err:
        st.warning(err)
    else:
        counts = pd.Series(ins["class_counts"], name="images")
        feats = pd.DataFrame(ins["features"]).T

        section("Training dataset at a glance", "Three image features that explain what the model learns.",
                "Data insights")
        small, big = counts.idxmin(), counts.idxmax()
        ratio = counts.max() / max(counts.min(), 1)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training images", f"{int(counts.sum()):,}")
        c2.metric("Classes", len(counts))
        c3.metric("Largest class", nice(big))
        c4.metric("Imbalance ratio", f"{ratio:.1f}x")

        # ---- Feature 1: class balance
        section("1 · Class balance", "How many training images exist for each class?")
        a, b = st.columns([1.6, 1], gap="large")
        with a:
            with st.container(border=True):
                class_bars(counts, "Images")
        with b:
            story(f"<b>{nice(small)}</b> has only {counts.min()} training images against {counts.max()} for "
                  f"<b>{nice(big)}</b> ({ratio:.1f}x more). The model sees the rare class less often, so "
                  "that is where recall is most at risk. This is why we report per-class precision and "
                  "recall and macro F1 instead of accuracy alone, and why new uploads of the rare class "
                  "are the most valuable for retraining.")

        if not feats.empty:
            # ---- Feature 2: colour
            section("2 · Colour", "How green is the leaf, and how do the RGB channels shift with disease?")
            a, b = st.columns([1.6, 1], gap="large")
            g_hi, g_lo = feats["green_ratio"].idxmax(), feats["green_ratio"].idxmin()
            with a:
                with st.container(border=True):
                    st.markdown("**Share of green pixels**")
                    class_bars(feats["green_ratio"], "Green share", fmt=".0%", height=240)
                with st.container(border=True):
                    st.markdown("**Mean RGB channel value**")
                    grouped_bars(feats[["red", "green", "blue"]],
                                 {"red": "#D2553F", "green": "#3E9A5B", "blue": "#4A78C2"},
                                 "Mean value (0–255)", fmt=".0f", height=240)
            with b:
                story(f"<b>{nice(g_hi)}</b> leaves have the highest share of green "
                      f"({feats.loc[g_hi, 'green_ratio']:.1%}) and <b>{nice(g_lo)}</b> the lowest "
                      f"({feats.loc[g_lo, 'green_ratio']:.1%}). Disease replaces green chlorophyll with "
                      "brown, grey or rust-coloured tissue, so loss of green is a direct signal of infection "
                      "that the model can learn from colour alone.")

            # ---- Feature 3: brightness & contrast
            section("3 · Brightness and contrast", "How patchy is the leaf surface?")
            a, b = st.columns([1.6, 1], gap="large")
            c_hi, c_lo = feats["contrast"].idxmax(), feats["contrast"].idxmin()
            with a:
                with st.container(border=True):
                    grouped_bars(feats[["brightness", "contrast"]],
                                 {"brightness": "#E0A100", "contrast": "#5B4BB7"}, "Grey level", fmt=".0f")
            with b:
                story(f"<b>{nice(c_hi)}</b> images have the highest contrast ({feats.loc[c_hi, 'contrast']:.1f}) "
                      f"and <b>{nice(c_lo)}</b> the lowest ({feats.loc[c_lo, 'contrast']:.1f}). Contrast measures "
                      "how much the grey level varies inside one photo: lesions and pustules create light and "
                      "dark patches, while a clean leaf is more uniform. Classes with similar colour AND similar "
                      "contrast are the ones the model confuses most, which you can check in the confusion "
                      "matrix on the Monitoring tab.")

    if metrics and metrics.get("history", {}).get("loss"):
        section("Training curves of the current model",
                "Training vs validation. Curves that stay close mean the model is not overfitting.", "Training")
        h = pd.DataFrame(metrics["history"])
        c1, c2 = st.columns(2, gap="large")
        with c1:
            with st.container(border=True):
                st.markdown("**Loss**")
                training_curve(h, ["loss", "val_loss"], "Loss")
        with c2:
            with st.container(border=True):
                st.markdown("**Accuracy**")
                training_curve(h, ["accuracy", "val_accuracy"], "Accuracy")

# ===================================================================== retrain
with tab_retrain:
    section("Keep the model learning", "Add new labelled field photos, then retrain. The new model only "
            "replaces the current one if it scores better on the test set.", "Retraining pipeline")
    html_block(f"""
    <div class="mg-steps">
      <div class="mg-step"><div class="n">{svg("upload", 16)}</div><div><div class="s">Upload</div><div class="d">Labelled images saved to the database</div></div></div>
      <div class="mg-step"><div class="n">{svg("sliders", 16)}</div><div><div class="s">Preprocess</div><div class="d">Validated, resized and normalised</div></div></div>
      <div class="mg-step"><div class="n">{svg("cpu", 16)}</div><div><div class="s">Retrain</div><div class="d">Fine-tunes the current model</div></div></div>
      <div class="mg-step"><div class="n">{svg("shield", 16)}</div><div><div class="s">Evaluate &amp; promote</div><div class="d">Kept only if accuracy improves</div></div></div>
    </div>""")

    up_col, queue_col = st.columns([1.2, 1], gap="large")

    # ---- Step 1: upload
    with up_col:
        with st.container(border=True):
            st.markdown("**Step 1 · Upload new labelled images**")
            st.caption("Select many images of ONE class, or upload a .zip with one folder per class "
                       f"({', '.join(CLASSES)}).")
            label = st.selectbox("Class of these images", CLASSES, format_func=nice)
            new_files = st.file_uploader("Images or a zip file", type=["jpg", "jpeg", "png", "zip"],
                                         accept_multiple_files=True, key="bulk_files")
            if new_files:
                st.caption(f"{len(new_files)} file(s) selected · "
                           f"{sum(f.size for f in new_files) / 1024 / 1024:.1f} MB")
            if icon_button("upload", "Upload to database", disabled=not new_files, use_container_width=True):
                with st.spinner("Validating and saving..."):
                    res, err = api("POST", "/upload", data={"label": label}, timeout=300,
                                   files=[("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                                          for f in new_files])
                if err:
                    st.error(err)
                else:
                    st.success(f"Saved {res['saved']} images to the database.")
                    st.toast(f"{res['saved']} images saved", icon=":material/check_circle:")
                    if res["rejected"]:
                        st.warning(f"Rejected {len(res['rejected'])} files")
                        rej = pd.DataFrame(res["rejected"])
                        html_table([[esc(v) for v in r] for r in rej.itertuples(index=False)],
                                   [esc(str(c).replace("_", " ").capitalize()) for c in rej.columns])
                    if res["auto_retrain_started"]:
                        st.info("Upload threshold reached: retraining started automatically.", icon=":material/autorenew:")

    # ---- queue + Step 2: trigger
    stats_now, _ = api("GET", "/stats", timeout=10)
    runs, _ = api("GET", "/retrain/status", timeout=10)
    running = bool(runs and runs.get("runs") and runs["runs"][0]["status"] == "running")

    with queue_col:
        with st.container(border=True):
            st.markdown("**Images waiting for retraining**")
            if stats_now:
                pend = stats_now["pending_uploads"]
                total, thr = sum(pend.values()), stats_now["retrain_threshold"]
                st.progress(min(total / thr, 1.0) if thr else 0.0,
                            text=f"{total} of {thr} images · automatic retraining starts at {thr}")
                if total:
                    class_bars(pd.Series(pend), "Waiting", height=200, wrap=True)
                else:
                    st.caption("The queue is empty. Upload images to fill it.")
            else:
                st.caption("Queue information unavailable.")

        with st.container(border=True):
            st.markdown("**Step 2 · Trigger retraining**")
            st.caption("Predictions keep working with the current model while retraining runs.")
            if icon_button("retrain", "Retrain model now", type="primary", use_container_width=True, disabled=running,
                         help="A run is already in progress." if running else None):
                res, err = api("POST", "/retrain")
                if err:
                    st.error(err)
                else:
                    st.success(f"Retraining run #{res['run_id']} started on {res['images']} new images.")
                    st.toast(f"Run #{res['run_id']} started", icon=":material/autorenew:")
                    runs, _ = api("GET", "/retrain/status", timeout=10)
                    running = bool(runs and runs.get("runs") and runs["runs"][0]["status"] == "running")

    # ---- Step 3: follow the run
    section("Step 3 · Follow the run", "History of retraining runs, newest first.")
    head_l, head_r = st.columns([4, 1])
    with head_r:
        icon_button("refresh", "Refresh status", key="refresh_runs", use_container_width=True)

    if runs and runs["runs"]:
        df = pd.DataFrame(runs["runs"])
        latest = df.iloc[0]
        if running:
            st.info("Retraining is running. Predictions keep working with the current model.", icon=":material/hourglass_top:")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Latest run", f"#{latest['id']}")
        c2.metric("Status", str(latest["status"]).capitalize())
        before, after = latest.get("accuracy_before"), latest.get("accuracy_after")
        c3.metric("Accuracy before", f"{before:.3f}" if pd.notna(before) else "—")
        c4.metric("Accuracy after", f"{after:.3f}" if pd.notna(after) else "—",
                  delta=f"{after - before:+.3f}" if pd.notna(before) and pd.notna(after) else None)

        def status_badge(v):
            v = str(v).lower()
            col = T["warn"] if v == "running" else T["bad"] if v in ("failed", "error") else T["ok"]
            return badge(v.capitalize(), col)

        def num(v):
            return "—" if v is None or pd.isna(v) else f"{float(v):.3f}"

        check = f'<span style="color:{T["ok"]}">{svg("check", 16)}</span>'
        cols = [("id", "Run", lambda v: f"#{esc(v)}"), ("status", "Status", status_badge),
                ("triggered_by", "Trigger", esc), ("started_at", "Started", esc), ("finished_at", "Finished", esc),
                ("n_images", "New images", esc), ("accuracy_before", "Acc. before", num),
                ("accuracy_after", "Acc. after", num),
                ("promoted", "Promoted", lambda v: check if bool(v) else f'<span style="color:{T["muted"]}">—</span>'),
                ("message", "Message", esc)]
        cols = [c for c in cols if c[0] in df]
        html_table([[f(r[k]) for k, _, f in cols] for _, r in df.iterrows()], [h for _, h, _ in cols])
    else:
        empty_state("folder", "No retraining runs yet", "Upload images and press <b style='display:inline'>Retrain model now</b>.")