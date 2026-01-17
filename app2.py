import streamlit as st
from model import tokenize, detokenize, predict, distribute_balls_min_one

# -----------------------------
# Session State Initialisierung
# -----------------------------
if "prompt_text" not in st.session_state:
    st.session_state.prompt_text = ""
if "generated_tokens" not in st.session_state:
    st.session_state.generated_tokens = []
if "current_ranges" not in st.session_state:
    st.session_state.current_ranges = []
if "highlight_token_id" not in st.session_state:
    st.session_state.highlight_token_id = None
if "norm_probs" not in st.session_state:
    st.session_state.norm_probs = []
if "model_probs" not in st.session_state:
    st.session_state.model_probs = []
if "balls_per_token" not in st.session_state:
    st.session_state.balls_per_token = []

st.title("LLM Token-Sack-Spiel 🎲")

# -----------------------------
# Start Prompt Eingabe
# -----------------------------
if st.session_state.prompt_text == "":
    start_prompt = st.text_input("Start-Prompt eingeben:")
    if start_prompt:
        st.session_state.prompt_text = start_prompt
        # Start-Prompt direkt in aktuellen Text
        st.session_state.generated_tokens.append(tokenize(start_prompt))
        # direkt erste Top-10 Tokens generieren
        predictions = predict(tokenize(st.session_state.prompt_text), top_k=10)
        token_ids = [tid for tid, _ in predictions]
        probs = [p for _, p in predictions]
        total_p = sum(probs)
        norm_probs = [p / total_p for p in probs]
        balls_per_token = distribute_balls_min_one(norm_probs, total_balls=100)
        st.session_state.norm_probs = norm_probs
        st.session_state.model_probs = probs
        st.session_state.balls_per_token = balls_per_token

        # Zahlenbereiche zuweisen
        start = 1
        ranges = []
        for tid, balls in zip(token_ids, balls_per_token):
            end = start + balls - 1
            ranges.append((start, end, tid))
            start = end + 1
        st.session_state.current_ranges = ranges
else:
    st.subheader("Start-Prompt:")
    st.text(st.session_state.prompt_text)

# -----------------------------
# Aktueller Text
# -----------------------------
st.subheader("Aktueller Text:")
display_text = ""
for idx, tid in enumerate(st.session_state.generated_tokens):
    # nur zuletzt gewähltes Token grün markieren
    if st.session_state.highlight_token_id == tid and idx == len(st.session_state.generated_tokens)-1:
        display_text += f"<span style='color:green;font-weight:bold'>{detokenize(tid)}</span>"
    else:
        display_text += detokenize(tid)
st.markdown(display_text, unsafe_allow_html=True)

# -----------------------------
# Top-10 Tabelle + Token-Buttons
# -----------------------------
if st.session_state.current_ranges:
    st.subheader("Top-10 Token Predictions:")

    # Tabellenüberschrift
    cols = st.columns([1, 1, 1, 1, 1, 1, 1])
    headers = ["Nr", "Token", "Modell-P", "Norm-P", "Anzahl Plättchen", "Zahlen-bereich", "Aktion"]
    for col, header in zip(cols, headers):
        col.markdown(f"**{header}**")

    # Tabelle Zeilen
    for idx, (start_num, end_num, tid) in enumerate(st.session_state.current_ranges):
        cols = st.columns([1, 1, 1, 1, 1, 1, 1])
        cols[0].write(idx + 1)
        cols[1].write(detokenize(tid))
        cols[2].write(f"{st.session_state.model_probs[idx]:.3f}")
        cols[3].write(f"{st.session_state.norm_probs[idx]:.3f}")
        cols[4].write(st.session_state.balls_per_token[idx])
        cols[5].write(f"{start_num}-{end_num}")
        if cols[6].button("wählen", key=f"{tid}"):
            # Token hinzufügen
            st.session_state.generated_tokens.append(tid)
            st.session_state.highlight_token_id = tid
            st.session_state.current_ranges = []  # Tabelle verschwinden lassen
            st.rerun()  # neuer Standard in aktueller Streamlit-Version

# -----------------------------
# Neue Token vorhersagen
# -----------------------------
if not st.session_state.current_ranges:
    if st.button("Neue Tokens vorhersagen"):
        predictions = predict(
            tokenize(st.session_state.prompt_text + "".join([detokenize(tid) for tid in st.session_state.generated_tokens])),
            top_k=10
        )
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
        st.session_state.highlight_token_id = None
        st.rerun()

# -----------------------------
# Neustarten Button
# -----------------------------
st.divider()
if st.button("🔄 Neustarten"):
    st.session_state.prompt_text = ""
    st.session_state.generated_tokens = []
    st.session_state.current_ranges = []
    st.session_state.highlight_token_id = None
    st.session_state.norm_probs = []
    st.session_state.model_probs = []
    st.session_state.balls_per_token = []
    st.rerun()
