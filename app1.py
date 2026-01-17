import streamlit as st
from model import tokenize, detokenize, predict

# -----------------------------
# Session State
# -----------------------------
if "start_prompt" not in st.session_state:
    st.session_state.start_prompt = ""
if "generated_tokens" not in st.session_state:
    st.session_state.generated_tokens = []
if "current_predictions" not in st.session_state:
    st.session_state.current_predictions = []
if "highlight_token_index" not in st.session_state:
    st.session_state.highlight_token_index = None
if "started" not in st.session_state:
    st.session_state.started = False
if "waiting_for_accept" not in st.session_state:
    st.session_state.waiting_for_accept = True

TOP_K = 10
MAX_TOKENS = 100

st.title("Token-Vorhersage-Spiel 🤖")

# -----------------------------
# Start-Prompt + Start-Button
# -----------------------------
if not st.session_state.started:
    start_prompt_input = st.text_input("Start-Prompt eingeben:")

    if st.button("▶️ Spiel starten") and start_prompt_input.strip():
        st.session_state.start_prompt = start_prompt_input
        st.session_state.generated_tokens = tokenize(start_prompt_input)
        st.session_state.started = True

        st.session_state.current_predictions = predict(
            st.session_state.generated_tokens, top_k=TOP_K
        )
        st.session_state.waiting_for_accept = True
        st.rerun()

else:
    st.subheader("Start-Prompt")
    st.text(st.session_state.start_prompt)

# -----------------------------
# Aktueller Text
# -----------------------------
if st.session_state.started:
    st.subheader("Aktueller Text")

    rendered = ""
    for i, tid in enumerate(st.session_state.generated_tokens):
        if i == st.session_state.highlight_token_index:
            rendered += (
                f"<span style='color:green;font-weight:bold'>"
                f"{detokenize(tid)}</span>"
            )
        else:
            rendered += detokenize(tid)

    st.markdown(rendered, unsafe_allow_html=True)

# -----------------------------
# Tabelle: Modell-Vorschläge
# -----------------------------
if st.session_state.started and st.session_state.current_predictions:
    st.subheader("Vorschläge des Modells")

    cols = st.columns([1, 3, 2])
    cols[0].markdown("**Nr.**")
    cols[1].markdown("**Token**")
    cols[2].markdown("**Modell-P**")

    for i, (token_id, prob) in enumerate(st.session_state.current_predictions):
        cols = st.columns([1, 3, 2])
        cols[0].write(i + 1)
        cols[1].write(detokenize(token_id))
        cols[2].write(f"{prob:.3f}")

# -----------------------------
# Button 1: Token übernehmen
# -----------------------------
if (
    st.session_state.started
    and st.session_state.waiting_for_accept
    and len(st.session_state.generated_tokens) < MAX_TOKENS
):
    if st.button("➡️ Nächstes Token übernehmen"):
        chosen_token_id = st.session_state.current_predictions[0][0]

        st.session_state.generated_tokens.append(chosen_token_id)
        st.session_state.highlight_token_index = len(st.session_state.generated_tokens) - 1
        st.session_state.waiting_for_accept = False
        st.rerun()

# -----------------------------
# Button 2: Neue Tokens vorhersagen
# -----------------------------
if (
    st.session_state.started
    and not st.session_state.waiting_for_accept
):
    if st.button("🔄 Neue Token vorhersagen"):
        st.session_state.current_predictions = predict(
            st.session_state.generated_tokens, top_k=TOP_K
        )
        st.session_state.highlight_token_index = None
        st.session_state.waiting_for_accept = True
        st.rerun()

# -----------------------------
# Neustart
# -----------------------------
st.divider()
if st.button("🧹 Neustarten"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
