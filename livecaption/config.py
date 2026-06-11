"""Default configuration constants.

All tunable parameters live here so they can be adjusted after real-world testing.
"""

from __future__ import annotations

# ---- Default models ----
# ASR: mlx-audio runs the nemotron-3.5 streaming transducer (Apple GPU / MLX; ~1.2GB in bf16).
# To save memory, switch to the 8bit quantized build
# "mlx-community/nemotron-3.5-asr-streaming-0.6b-8bit".
DEFAULT_ASR_MODEL = "mlx-community/nemotron-3.5-asr-streaming-0.6b"
# Translation: Hy-MT2 (Tencent Hunyuan MT, 3rd gen) 1.8B 8bit, ~2GB memory
DEFAULT_MT_MODEL = "mlx-community/Hy-MT2-1.8B-8bit"

# ---- Audio ----
SAMPLE_RATE = 16000  # ASR / VAD / audiotee all use 16k mono

# ---- ASR (mlx-audio nemotron-3.5) ----
# English by default; CLI --asr-lang overrides it (the model supports 40 locales, and an
# invalid value lists all options).
# "auto" lets the model detect the language, but with mixed Chinese/English in meetings the
# detected language jumps around, so it's not recommended.
ASR_LANGUAGE = "en-US"
# Streaming look-ahead [left, right]: right+1 downsampled frames = how big a chunk is fed at
# once = partial refresh granularity.
# Settings the model was trained on: [56,0]=80ms  [56,3]=320ms  [56,6]=560ms  [56,13]=1120ms
# (best accuracy)
ASR_ATT_CONTEXT = [56, 6]
# two-pass correction: at finalization, re-decode the whole sentence with the largest
# look-ahead; final (and the translation input) use the re-decoded result -- partial trades
# for speed (each frame sees only 480ms of future), final re-decodes for accuracy (each frame
# sees 1.12s of future).
ASR_TWO_PASS = True
ASR_FINAL_ATT_CONTEXT = [56, 13]

# ---- Speaker diarization (Sortformer v2.1 streaming; disable with --no-diarize) ----
# At finalization the whole sentence is fed at once (the model's native operating point is 15s
# chunks; feeding small streaming chunks makes speaker identity drift), then text is split and
# attributed to each speaker by RNNT token timestamps, with a separate final + translation per
# segment.
DIAR_MODEL = "mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16"
DIAR_THRESHOLD = 0.5  # frame-level (80ms) speaker-activity probability threshold
# how often to do a read-only probe of the current sentence, giving the partial a tentative
# speaker label
DIAR_PEEK_SEC = 2.0
# Smoothing: a speaker turn shorter than this (seconds) inside one utterance is absorbed into
# the surrounding speaker instead of becoming its own segment. Makes diarization less twitchy --
# a brief probability blip or a one-word backchannel won't split one speaker's sentence into a
# fragment, and since each speaker segment becomes a separate final, it also yields fewer tiny
# finals. Raise it to be stickier; set 0 to honor every momentary speaker change.
DIAR_MIN_SEGMENT_SEC = 1.0

# ---- VAD / endpoint detection (Silero VAD mlx build, one speech probability every 32ms) ----
VAD_MODEL = "mlx-community/silero-vad"
VAD_THRESHOLD = 0.5  # speech-probability threshold; >= counts as a speech frame
# audio padded back in when speech onset is detected, to keep the first word from being clipped
VAD_PRE_ROLL_MS = 320
# Three rules in an OR relationship; any one satisfied marks the end of a sentence (following
# sherpa-onnx semantics). All values are in seconds.
# trailing-silence threshold when there is no decoded text yet: only resets, does not emit a final
RULE1_MIN_TRAILING_SILENCE = 2.4
# rule2 is punctuation-aware. The nemotron RNNT emits . ? ! inline, so we let "is the
# sentence already complete?" decide how long a pause must be before we cut:
#   - text already ends with . ? ! -> a short pause is enough (the boundary lands on a real
#     sentence end instead of mid-word)
#   - no terminal punctuation yet -> wait the longer silence, so a mid-sentence thinking
#     pause doesn't fragment one utterance into pieces
RULE2_PUNCT_SILENCE = 0.6  # trailing silence to cut when the text ends a sentence
RULE2_MIN_TRAILING_SILENCE = 1.2  # trailing silence to cut when it does not
RULE3_MIN_UTTERANCE_LENGTH = 20.0  # max seconds before a sentence is force-cut

# ---- Translation (Hy-MT2's officially recommended prompt and sampling parameters) ----
TRANSLATE_PROMPT = (
    "Translate the following text into {target_lang}. "
    "Note that you should only output the translated result "
    "without any additional explanation:\n\n{text}"
)
# Translation with context: uses Hy-MT2's official "background information" template verbatim
# (straight from the README, unmodified).
# Preceding sentences serve as background so the model can resolve references / terminology /
# topic.
# number of preceding sentences attached when translating (default 3 for better coherence;
# 0 = sentence-by-sentence with no context)
MT_CONTEXT_SENTENCES = 3
# Live translation preview: while a sentence is still being spoken, show a provisional ZH line
# under the EN partial, refreshed once the in-progress text has grown by this many words AND
# contains a sentence-ender (.?!). Debounced so we don't retranslate every chunk; the final
# two-pass-corrected re-translation still replaces it. Set 0 to disable the live preview.
MT_PREVIEW_MIN_WORDS = 6
TRANSLATE_PROMPT_WITH_CONTEXT = (
    "[Background Information]\n"
    "{context}\n\n"
    "Please translate the following text into {target_lang}, "
    "taking the provided background information into consideration.\n\n"
    "[Source Text]\n{text}"
)
MT_TEMPERATURE = 0.7
MT_TOP_P = 0.6
MT_TOP_K = 20
MT_REPETITION_PENALTY = 1.05
# plenty for a single subtitle line; just an upper-bound safeguard against the model running away
MT_MAX_TOKENS = 256
DEFAULT_TARGET_LANG = "Chinese"
