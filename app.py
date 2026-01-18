import streamlit as st
import streamlit.components.v1 as components
from model import tokenize, detokenize, predict, distribute_balls_threshold, add_token, current_token_ids, render_token_html, reset_app

# =====================================================
# Konfiguration
# =====================================================
TOP_K = 10
TOTAL_BALLS = 100
MIN_NORM_PROB_FOR_BALL = 0.005  # 0.5%

TOKEN_COLORS = [
    "#ffe066",  # gelb
    "#b2f2bb",  # grün
    "#a5d8ff",  # blau
    "#ffd6a5",  # orange
    "#f3d9fa",  # lila
    "#ffadad",  # rot
    "#e7f5ff",  # hellblau
    "#fff3bf",  # sand
    "#d0ebff",  # eisblau
    "#e6fcf5",  # mint
]

# =====================================================
# Session State Initialisierung
# =====================================================
def init_state():
    defaults = {
        "mode": None,
        "started": False,
        "token_strings": [],
        "token_ids": [],
        "color_index": 0,
        "current_predictions": [],
        "ranges": [],
        "model_probs": [],
        "norm_probs": [],
        "balls": [],
        "show_token_ids": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# =====================================================
# UI
# =====================================================
st.title("Wort Vorhersager 🔮 ")
st.subheader("Sprachmodell: german-gpt2-medium")


# -------------------------
# Modus wählen
# -------------------------
if st.session_state.mode is None:
    st.subheader("Modus wählen")
    col1, col2 = st.columns(2)
    if col1.button("Deterministisch", key="btn_mode_deterministic"):
        st.session_state.mode = "deterministic"
        st.rerun()
    if col2.button("Probabilistisch", key="btn_mode_probabilistic"):
        st.session_state.mode = "probabilistic"
        st.rerun()
    st.stop()

# -------------------------
# Startprompt
# -------------------------
if not st.session_state.started:
    def _do_start():
        prompt_val = st.session_state.get("prompt", "").strip()
        if prompt_val:
            for tid in tokenize(prompt_val):
                add_token(tid)
            st.session_state.started = True

    # Enter löst on_change aus und startet
    st.text_input("Start-Prompt eingeben:", key="prompt", on_change=_do_start)

    # Button bleibt als Alternative
    if st.button("▶️ Starten", key="btn_start"):
        _do_start()
        st.rerun()

    st.stop()

# -------------------------
# Aktueller Text mit Hover
# -------------------------
st.subheader("Aktueller Text")

# Text-HTML generieren
text_html = ""
for i, tok in enumerate(st.session_state.token_strings):
    color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
    text_html += render_token_html(tok, color, is_text=True, index=i)

# Token-ID-HTML (für bidirektionales Hover immer mitliefern, initial ggf. versteckt)
ids_html = ""
for i, tid in enumerate(st.session_state.token_ids):
    color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
    ids_html += render_token_html(tid, color, is_text=False, index=i)

# Kombiniertes HTML + funktionierendes JS via components.html
# Toggle-Button zwischen Text und Token-ID-Zeile
# Schriftart: system-ui Stack, damit sie zur restlichen App passt.
initial_display = "block" if st.session_state.show_token_ids else "none"
initial_label = "Token-IDs ausblenden" if st.session_state.show_token_ids else "Token-IDs anzeigen"

hover_html = f"""
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<style>
  :root {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Helvetica Neue, Arial, sans-serif;
    font-size: 1rem;
  }}
  body {{
    margin: 0;
    padding: 0;
    font-family: inherit;
    font-size: inherit;
  }}
  #tokens-container {{
    font-family: inherit;
    font-size: inherit;
  }}
  #text-line, #ids-line {{
    font-family: inherit;
    font-size: inherit;
  }}
  .token {{
    font-family: inherit !important;
    font-size: inherit !important;
    transition: opacity 120ms ease-in-out;
  }}
  #toggle-row {{
    margin-top: 12px;
    margin-bottom: 12px;
  }}
  #toggle-btn {{
    appearance: none;
    border: 1px solid rgba(49, 51, 63, 0.2);
    border-radius: 0.5rem;
    padding: 0.35rem 0.6rem;
    background-color: rgba(240, 242, 246, 0.6);
    color: rgba(49, 51, 63, 1);
    font: inherit;
    cursor: pointer;
  }}
  #toggle-btn:hover {{
    background-color: rgba(240, 242, 246, 0.9);
  }}
</style>
</head>
<body>
<div id="tokens-container">
  <div id="text-line">{text_html}</div>

  <div id="toggle-row">
    <button id="toggle-btn" type="button" role="button">{initial_label}</button>
  </div>

  <div id="ids-line" style="margin-top: 0px; display: {initial_display};">{ids_html}</div>
</div>

<script>
(function() {{
  const textTokens = document.querySelectorAll('#text-line .token-text');
  const idTokens = document.querySelectorAll('#ids-line .token-id'); // vorhanden, auch wenn versteckt
  const idsLine = document.getElementById('ids-line');
  const toggleBtn = document.getElementById('toggle-btn');

  function setState(activeIdx) {{
    // Text-Tokens
    textTokens.forEach((t, i) => {{
      const baseColor = t.getAttribute('data-color');
      t.style.backgroundColor = baseColor;
      t.style.opacity = (i === activeIdx) ? '1' : '0.35';
    }});
    // ID-Tokens (falls vorhanden)
    idTokens.forEach((tidSpan, i) => {{
      const idColor = tidSpan.getAttribute('data-color');
      tidSpan.style.backgroundColor = idColor;
      tidSpan.style.opacity = (i === activeIdx) ? '1' : '0.35';
    }});
  }}

  function resetState() {{
    textTokens.forEach((t) => {{
      t.style.opacity = '1';
      t.style.backgroundColor = t.getAttribute('data-color');
    }});
    idTokens.forEach((tidSpan) => {{
      tidSpan.style.opacity = '1';
      tidSpan.style.backgroundColor = tidSpan.getAttribute('data-color');
    }});
  }}

  // Hover über Text
  textTokens.forEach((tok, idx) => {{
    tok.addEventListener('mouseenter', () => setState(idx));
    tok.addEventListener('mouseleave', resetState);
  }});

  // Bidirektionales Hover: auch über Token-IDs
  idTokens.forEach((tok, idx) => {{
    tok.addEventListener('mouseenter', () => setState(idx));
    tok.addEventListener('mouseleave', resetState);
  }});

  // Toggle-Button Logik (zwischen Text und IDs)
  function updateToggleBtnLabel() {{
    const visible = idsLine.style.display !== 'none';
    toggleBtn.textContent = visible ? 'Token-IDs ausblenden' : 'Token-IDs anzeigen';
  }}

  toggleBtn.addEventListener('click', () => {{
    const visible = idsLine.style.display !== 'none';
    idsLine.style.display = visible ? 'none' : 'block';
    updateToggleBtnLabel();
  }});

  updateToggleBtnLabel();
}})();
</script>
</body>
</html>
"""

# Höhe: etwas größer, damit beide Zeilen bequem passen, inkl. Toggle.
components.html(hover_html, height=200, scrolling=False)

# =====================================================
# MODUS A – Deterministisch
# =====================================================
if st.session_state.mode == "deterministic":
    # Vorab Vorhersagen berechnen
    preds = predict(current_token_ids(), top_k=TOP_K)

    # Überschrift + Weiter-Button in einer Zeile
    header_cols = st.columns([4, 1])
    with header_cols[0]:
        st.subheader("Modell-Vorschläge")
    with header_cols[1]:
        if st.button("➡️ Weiter", key="btn_next_det"):
            if preds:
                add_token(preds[0][0])
                st.rerun()

    # Tabelle mit zusätzlicher Spalte "Token-ID" und "Wahrscheinlichkeit" in %
    cols = st.columns([1, 2, 3, 2])  # Nr, Token-ID, Token, Wahrscheinlichkeit
    cols[0].markdown("**Nr**")
    cols[1].markdown("**Token-ID**")
    cols[2].markdown("**Token**")
    cols[3].markdown("**Wahrscheinlichkeit**")

    for i, (tid, p) in enumerate(preds):
        row = st.columns([1, 2, 3, 2])
        row[0].write(i + 1)
        row[1].write(tid)
        row[2].write(detokenize(tid))
        row[3].write(f"{p * 100:.1f} %")

# =====================================================
# MODUS B – Probabilistisch
# =====================================================
if st.session_state.mode == "probabilistic":
    if not st.session_state.ranges:
        # Top-K holen und normieren
        preds = predict(current_token_ids(), top_k=TOP_K)
        token_ids = [t for t, _ in preds]
        probs = [p for _, p in preds]

        s = sum(probs)
        norm = [p / s for p in probs] if s > 0 else [0.0] * len(probs)

        # Neue Verteilung mit Schwellenwert
        balls = distribute_balls_threshold(norm, TOTAL_BALLS, MIN_NORM_PROB_FOR_BALL)

        # Bereiche berechnen: Tokens mit 0 Plättchen bekommen "-" und werden nicht in den Start-End-Zähler einbezogen
        ranges = []
        start = 1
        for tid, b in zip(token_ids, balls):
            if b > 0:
                end = start + b - 1
                ranges.append((start, end, tid))
                start = end + 1
            else:
                ranges.append((None, None, tid))

        st.session_state.ranges = ranges
        st.session_state.model_probs = probs
        st.session_state.norm_probs = norm
        st.session_state.balls = balls

    st.subheader("Token-Sack")
    headers = ["Nr", "Token-ID", "Token", "Modell-P", "Norm-P", "Plättchen", "Bereich", "Aktion"]
    cols = st.columns([1, 2, 2, 2, 2, 2, 2, 2])
    for c, h in zip(cols, headers):
        c.markdown(f"**{h}**")

    for i, (start, end, tid) in enumerate(st.session_state.ranges):
        cols = st.columns([1, 2, 2, 2, 2, 2, 2, 2])
        cols[0].write(i + 1)
        cols[1].write(tid)
        cols[2].write(detokenize(tid))
        cols[3].write(f"{st.session_state.model_probs[i]:.3f}")
        cols[4].write(f"{st.session_state.norm_probs[i]:.3f}")
        # Plättchen und Bereich anzeigen
        b = st.session_state.balls[i]
        cols[5].write(b if b > 0 else "-")
        cols[6].write(f"{start}-{end}" if (start is not None and end is not None) else "-")
        # Ziehen-Button nur, wenn Plättchen > 0
        can_pick = b > 0
        if cols[7].button("ziehen", key=f"btn_pick_{i}", disabled=not can_pick):
            if can_pick:
                add_token(tid)
                st.session_state.ranges = []
                st.rerun()

# -------------------------
# Neustart
# -------------------------
st.divider()
if st.button("🔄 Neustarten", key="btn_reset"):
    reset_app()