import logging
import os
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

MODEL_NAME = "dbmdz/german-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.to(DEVICE)
model.eval()

# Optionales Pipeline-Objekt (nur für andere Generierung)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1  # erzwingt CPU
)

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
# Optional: Textgenerierung über pipeline
# -----------------------------------

def generate_text(text: str, num_tokens: int) -> str:
    """
    Generate text using the HuggingFace pipeline.
    """
    output = pipe(
        text,
        max_new_tokens=num_tokens,
        do_sample=True,
        temperature=1.0
    )
    return output[0]["generated_text"]


def distribute_balls_min_one(probabilities, total_balls=100):
    """
    probabilities: list of floats, sollte auf 1 normiert sein
    returns: list of ints, sum == total_balls
    ensures: each token gets mindestens 1 Kugel
    """
    n = len(probabilities)

    # Sicherheitsfallback für NaNs oder negative Werte
    if any(p != p or p < 0 for p in probabilities):
        probabilities = [1 / n] * n

    raw = [p * total_balls for p in probabilities]

    floors = [max(1, int(x)) for x in raw]
    remainder = total_balls - sum(floors)

    # Größte Reste verteilen
    remainders = [(i, raw[i] - floors[i]) for i in range(n)]
    remainders.sort(key=lambda x: x[1], reverse=True)

    i = 0
    while remainder > 0:
        floors[remainders[i % n][0]] += 1
        remainder -= 1
        i += 1

    # Falls zu viele Kugeln verteilt (sehr selten)
    i = 0
    while remainder < 0:
        if floors[i] > 1:
            floors[i] -= 1
            remainder += 1
        i = (i + 1) % n

    return floors

