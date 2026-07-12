import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def cos_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def grounding_score(answer, context_chunks):
    avec = model.encode([answer])[0]
    cvecs = model.encode(context_chunks)
    return max(cos_sim(avec, c) for c in cvecs)

STOPWORDS = {"the", "a", "an", "is", "was", "in", "on", "of", "and", "to", "by", "at"}

def keyword_overlap(answer, context_chunks):
    answer_words = {w.lower().strip(".,") for w in answer.split()} - STOPWORDS
    context_words = set()
    for c in context_chunks:
        context_words |= {w.lower().strip(".,") for w in c.split()} - STOPWORDS
    if not answer_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)

def is_grounded(answer, context_chunks, sim_threshold=0.5, overlap_threshold=0.4):
    sim = grounding_score(answer, context_chunks)
    overlap = keyword_overlap(answer, context_chunks)
    return sim >= sim_threshold and overlap >= overlap_threshold

context = [
    "The Eiffel Tower is located in Paris, France, and was completed in 1889.",
    "Python was created by Guido van Rossum and released in 1991.",
]

good_answer = "The Eiffel Tower was completed in 1889 in Paris."
bad_answer = "The Eiffel Tower was built by aliens in the year 3000 on Mars."

good_score = grounding_score(good_answer, context)
bad_score = grounding_score(bad_answer, context)

print("good:", good_score, is_grounded(good_answer, context))
print("bad:", bad_score, is_grounded(bad_answer, context))

assert is_grounded(good_answer, context), "known-good answer should pass"
assert not is_grounded(bad_answer, context), "known-bad answer should be flagged"
print("OK")
