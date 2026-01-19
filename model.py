import logging
import os
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# -----------------------------------
# Logging/Umgebungsvariablen
# -----------------------------------

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

# -----------------------------------
# Gerät festlegen (nur CPU)
# -----------------------------------

DEVICE = torch.device("cpu")

# -----------------------------------
# Modell laden
# -----------------------------------

MODEL_NAME = "kkirchheim/german-gpt2-medium"
# MODEL_NAME = "dbmdz/german-gpt2-medium"
#   -> aktuell in Streamlit App verwendet!
# evtl noch folgende probieren:
#   MODEL_NAME = "tum-nlp/german-gpt2_easy"
#   MODEL_NAME = "EleutherAI/gpt-neo-125M"


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()


# -----------------------------------
# Tokenisierung
# -----------------------------------

def tokenize(text: str) -> list[int]:
    """
    Convert a string into a list of token IDs.
    """
    return tokenizer.encode(text, add_special_tokens=False)

def detokenize(token_ids) -> str:
    """
    Convert token ID(s) back into a string.
    Accepts either an int or a list of ints.
    """
    if isinstance(token_ids, int):
        token_ids = [token_ids]
    return tokenizer.decode(token_ids)

# -----------------------------------
# Top-k Vorhersage
# -----------------------------------

def predict(token_ids: list[int], top_k: int = 10):
    """
    Predict the top-k most likely next tokens given a list of token IDs.
    Returns: list of (token_id, probability).
    """
    input_ids = torch.tensor([token_ids]).to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    next_token_logits = logits[0, -1]
    probs = torch.softmax(next_token_logits, dim=-1)

    top_probs, top_indices = torch.topk(probs, top_k)

    return [
        (idx.item(), prob.item())
        for idx, prob in zip(top_indices, top_probs)
    ]

# -----------------------------------
# Normierung der Wahrscheinlichkeiten
# -----------------------------------

def distribute_balls_threshold(probabilities, total_balls=100, min_prob=0.005):
    """
    Verteilt total_balls auf Tokens mit p >= min_prob mittels Largest Remainder.
    Tokens unterhalb der Schwelle erhalten 0 Plättchen.
    Falls kein Token die Schwelle erreicht, wird ohne Schwelle verteilt.

    probabilities: list[float], Summe ~ 1 (normiert)
    returns: list[int] mit Summe == total_balls
    """
    n = len(probabilities)
    # Sicherheitsfallback: NaN/negativ -> Gleichverteilung
    if any((p != p) or (p < 0) for p in probabilities):  # NaN check via p != p
        probabilities = [1 / n] * n

    # Maske für Tokens über Schwelle
    eligible = [p >= min_prob for p in probabilities]
    if not any(eligible):
        # Fallback: ohne Schwelle verteilen
        eligible = [True] * n

    # Rohwerte und Floors
    raw = [p * total_balls if eligible[i] else 0.0 for i, p in enumerate(probabilities)]
    floors = [int(x) for x in raw]
    remainder = total_balls - sum(floors)

    # Largest Remainder Methode auf eligible Tokens
    remainders = [(i, raw[i] - floors[i]) for i in range(n) if eligible[i]]
    remainders.sort(key=lambda x: x[1], reverse=True)

    i = 0
    while remainder > 0 and remainders:
        idx = remainders[i % len(remainders)][0]
        floors[idx] += 1
        remainder -= 1
        i += 1

    # Sollte nie passieren (weil floors via floor <= total_balls), aber zur Sicherheit:
    i = 0
    while remainder < 0 and any(f > 0 for f in floors):
        if floors[i] > 0:
            floors[i] -= 1
            remainder += 1
        i = (i + 1) % n

    return floors


# =====================================================
# Hilfsfunktionen
# =====================================================
def add_token(token_id):
    tok = detokenize(token_id)
    st.session_state.token_strings.append(tok)
    st.session_state.token_ids.append(token_id)
    st.session_state.color_index += 1

def current_token_ids():
    return st.session_state.token_ids.copy()

def render_token_html(tok, color, is_text=True, index=None):
    """Generiert HTML für Text-Token oder Token-ID-Span mit Datenattributen."""
    safe_tok = tok.replace(" ", "&nbsp;") if is_text else str(tok)
    cls = "token token-text" if is_text else "token token-id"
    tid_attr = f"id='tid_{index}'" if not is_text and index is not None else ""
    padding_right = "0px" if is_text else "2px"
    margin_right = "0px" if is_text else "2px"

    html = (
        f"<span class='{cls}' {tid_attr} "
        f"data-index='{index}' data-color='{color}' "
        f"style='"
        f"background-color:{color};"
        f"color:black;"
        f"display:inline-flex;"
        f"align-items:center;"
        f"justify-content:center;"
        f"min-width:0.2em;"
        f"height:1.4em;"
        f"line-height:1.4em;"
        f"vertical-align:middle;"
        f"padding:0px {padding_right};"
        f"margin-right:{margin_right};"
        f"font-family:inherit;"
        f"font-size:inherit;"
        f"'>"
        f"{safe_tok}</span>"
    )
    return html

def reset_app():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def chip(value, color="#2e7d32"):
    return (
        f"<span style='display:inline-flex;align-items:center;"
        f"height:32px;min-height:32px;line-height:1;"
        f"padding:0 10px;border-radius:4px;"
        f"white-space:nowrap;"
        f"background:rgba(240,242,246,0.85);"
        f"color:{color};font-weight:600;font-family:inherit;'>"
        f"{value}</span>"
    )