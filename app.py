import streamlit as st
from model import tokenize, detokenize, predict, distribute_balls_min_one

# -----------------------------
# Session State Initialisierung
# -----------------------------
if "mode" not in st.session_state:
    st.session_state.mode = None  # "deterministic" oder "probabilistic"
if "started" not in st.session_state:
    st.session_state.started = False

# Allgemein
if "generated_tokens" not in st.session_state:
    st.session_state.generated_tokens = []
if "highlight_token_index" not in st.session_state:
    st.session_state.highlight_token_index = None

# Deterministisch
if "start_prompt" not in st.session_state:
    st.session_state.start_prompt = ""
if "current_predictions" not in st.session_state:
    st.session_state.current_predictions = []
if "waiting_for_accept" not in st.session_state:
    st.session_state.waiting_for_accept = True

# Probabilistisch
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""
if "current_ranges" not in st.session_state:
    st.session_state.current_ranges = []
if "norm_probs" not in st.session_state:
    st.session_state.norm_probs = []
if "model_probs" not in st.session_state:
    st.session_state.model_probs = []
if "balls_per_token" not in st.session_state:
    st.session_state.balls_per_token = []

TOP_K = 10
MAX_TOKENS = 100

st.title("Sprachmodell Token Predictor")

# -----------------------------
# Modus-Auswahl
# -----------------------------
if st.session_state.mode is None:
    st.subheader("Wähle einen Modus:")
    col1, col2 = st.columns(2)
    if col1.button("Simpel"):
        st.session_state.mode = "deterministic"
        st.rerun()
    if col2.button("Komplex"):
        st.session_state.mode = "probabilistic"
        st.rerun()

# -----------------------------
# Start-Prompt Eingabe (erst nach Moduswahl)
# -----------------------------
elif not st.session_state.started:
    if st.session_state.mode == "deterministic":
        prompt_input = st.text_input("Start-Prompt eingeben:")
        if st.button("▶️ Spiel starten") and prompt_input.strip():
            st.session_state.start_prompt = prompt_input
            st.session_state.generated_tokens = tokenize(prompt_input)
            st.session_state.current_predictions = predict(st.session_state.generated_tokens, top_k=TOP_K)
            st.session_state.waiting_for_accept = True
            st.session_state.started = True
            st.rerun()
    else:  # probabilistic
        prompt_input = st.text_input("Start-Prompt eingeben:")
        if st.button("▶️ Spiel starten") and prompt_input.strip():
            st.session_state.prompt_text = prompt_input
            st.session_state.generated_tokens = tokenize(prompt_input)
            # direkt erste Top-10 Tokens generieren
            predictions = predict(st.session_state.generated_tokens, top_k=10)
            token_ids = [tid for tid, _ in predictions]
            probs = [p for _, p in predictions]
            total_p = sum(probs)
            norm_probs = [p / total_p for p in probs]
            balls_per_token = distribute_balls_min_one(norm_probs, total_balls=100)
            st.session_state.norm_probs = norm_probs
            st.session_state.model_probs = probs
            st.session_state.balls_per_token = balls_per_token

            start = 1
            ranges = []
            for tid, balls in zip(token_ids, balls_per_token):
                end = start + balls - 1
                ranges.append((start, end, tid))
                start = end + 1
            st.session_state.current_ranges = ranges

            st.session_state.highlight_token_index = None
            st.session_state.started = True
            st.rerun()

# -----------------------------
# Aktueller Text
# -----------------------------
if st.session_state.started:
    st.subheader("Aktueller Text:")
    rendered = ""
    for i, tid in enumerate(st.session_state.generated_tokens):
        if st.session_state.mode == "deterministic":
            if i == st.session_state.highlight_token_index:
                rendered += f"<span style='color:green;font-weight:bold'>{detokenize(tid)}</span>"
            else:
                rendered += detokenize(tid)
        else:  # probabilistic
            if st.session_state.highlight_token_index is not None and i == st.session_state.highlight_token_index:
                rendered += f"<span style='color:green;font-weight:bold'>{detokenize(tid)}</span>"
            else:
                rendered += detokenize(tid)
    st.markdown(rendered, unsafe_allow_html=True)

# -----------------------------
# Modus-spezifische Anzeige
# -----------------------------

# Deterministisch
if st.session_state.started and st.session_state.mode == "deterministic":
    st.subheader("Vorschläge des Modells für das nächste Token:")
    cols = st.columns([1, 3, 2])
    cols[0].markdown("**Nr.**")
    cols[1].markdown("**Token**")
    cols[2].markdown("**Modell-P**")
    for i, (token_id, prob) in enumerate(st.session_state.current_predictions):
        cols = st.columns([1, 3, 2])
        cols[0].write(i + 1)
        cols[1].write(detokenize(token_id))
        cols[2].write(f"{prob:.3f}")

    # Buttons deterministisch
    if st.session_state.waiting_for_accept and len(st.session_state.generated_tokens) < MAX_TOKENS:
        if st.button("➡️ Token wählen"):
            chosen_token_id = st.session_state.current_predictions[0][0]
            st.session_state.generated_tokens.append(chosen_token_id)
            st.session_state.highlight_token_index = len(st.session_state.generated_tokens) - 1
            st.session_state.waiting_for_accept = False
            st.rerun()
    elif not st.session_state.waiting_for_accept:
        if st.button("🔄 Neue Token vorhersagen"):
            st.session_state.current_predictions = predict(st.session_state.generated_tokens, top_k=TOP_K)
            st.session_state.highlight_token_index = None
            st.session_state.waiting_for_accept = True
            st.rerun()

# Probabilistisch
elif st.session_state.started and st.session_state.mode == "probabilistic":
    st.subheader("Vorschläge des Modells für das nächste Token:")
    # Tabellenüberschrift
    cols = st.columns([1, 1, 1, 1, 1, 1, 1])
    headers = ["Nr", "Token", "Modell-P", "Norm-P", "Anzahl Plättchen", "Zahlen-bereich", "Aktion"]
    for col, header in zip(cols, headers):
        col.markdown(f"**{header}**")

    for idx, (start_num, end_num, tid) in enumerate(st.session_state.current_ranges):
        cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        cols[0].write(idx + 1)
        cols[1].write(detokenize(tid))
        cols[2].write(f"{st.session_state.model_probs[idx]:.3f}")
        cols[3].write(f"{st.session_state.norm_probs[idx]:.3f}")
        cols[4].write(st.session_state.balls_per_token[idx])
        cols[5].write(f"{start_num}-{end_num}")
        if cols[6].button("wählen", key=f"{tid}"):
            st.session_state.generated_tokens.append(tid)
            st.session_state.highlight_token_index = len(st.session_state.generated_tokens) - 1
            st.session_state.current_ranges = []
            st.rerun()

    # Button neue Token vorhersagen
    if not st.session_state.current_ranges:
        if st.button("🔄 Neue Tokens vorhersagen"):
            predictions = predict(st.session_state.generated_tokens, top_k=TOP_K)
            token_ids = [tid for tid, _ in predictions]
            probs = [p for _, p in predictions]
            total_p = sum(probs)
            norm_probs = [p / total_p for p in probs]
            balls_per_token = distribute_balls_min_one(norm_probs, total_balls=100)
            st.session_state.norm_probs = norm_probs
            st.session_state.model_probs = probs
            st.session_state.balls_per_token = balls_per_token

            start = 1
            ranges = []
            for tid, balls in zip(token_ids, balls_per_token):
                end = start + balls - 1
                ranges.append((start, end, tid))
                start = end + 1
            st.session_state.current_ranges = ranges
            st.session_state.highlight_token_index = None
            st.rerun()

# -----------------------------
# Neustarten
# -----------------------------
st.divider()
if st.button("🔄 Neustarten"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
