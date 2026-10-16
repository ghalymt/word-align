"""Where a subtitle may break -- between the lines of a cue and between cues.

:func:`break_cost` scores the break between two pieces of text: negative is
a good place to break, positive a bad one, in the same units as the
line-length imbalance the balancer also weighs (characters).

English follows the GoTranscript / broadcast rules; keep together:

* a subject pronoun and its verb ("we | wake up")
* auxiliaries, modals and the verb ("have | been", "doesn't | help")
* phrasal verbs and verb + object pronoun ("tuned | out", "enjoy | it")
* articles, determiners, possessives and what they modify ("the | overall")
* a preposition and its object ("to | politics")
* names and titles ("Henry | Schein", "Dr. | Smith")
* numbers and their units ("21 | patients", "15.4 | million tons")

and prefer to break after a sentence end, then after clause punctuation,
then before a conjunction, relative pronoun or preposition. Other
languages keep the original, multilingual word lists from ``config``.
"""
from __future__ import annotations

import re
from typing import Sequence

from .config import CLAUSE_MARKERS, NON_TERMINAL_ABBREVIATIONS, PREFER_NEW_LINE_WORDS

SENTENCE_END = -12.0   # "...okay. | Wonderful."
CLAUSE_END = -8.0      # "...okay, | so we..."
TAG_EDGE = -6.0        # next to a [sound tag]
GOOD_START = -5.0      # a conjunction or relative pronoun starts the line
PREP_START = -3.0      # a preposition starts the line
BAD = 15.0             # splits words that belong together
SOFT_BAD = 6.0
OBJECT_SPLIT = 12.0    # "enjoy | it"
TITLE_SPLIT = 20.0     # "Dr. | Smith"

_CLOSERS = "\"'”’»)]}"
_TERMINAL = (".", "?", "!", "…")
_CLAUSE = (",", ";", ":", "—", "–")

_ARTICLES = {"a", "an", "the"}
_PREPOSITIONS = {
    "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto",
    "about", "as", "like", "over", "under", "after", "before", "between",
    "through", "during", "without", "within", "against", "among", "around",
    "behind", "beyond", "toward", "towards", "upon", "via", "per", "across",
    "along", "near", "than", "since", "until", "despite", "throughout",
}
_CONJUNCTIONS = {
    "and", "but", "or", "nor", "yet", "because", "if", "when", "while",
    "although", "though", "whereas", "unless", "until", "whether", "since",
    "once", "so",
}
_RELATIVES = {"that", "which", "who", "whom", "whose", "where", "what", "how", "why"}
_AUXILIARIES = {
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did",
    "will", "would", "shall", "should", "can", "could", "may", "might", "must",
    "ought", "gonna", "wanna", "gotta",
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "cannot",
    "couldn't", "shouldn't", "mustn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "ain't",
    "i'm", "you're", "we're", "they're", "he's", "she's", "it's", "that's",
    "there's", "here's", "what's", "who's", "i've", "you've", "we've",
    "they've", "i'll", "you'll", "we'll", "they'll", "he'll", "she'll",
    "it'll", "that'll", "i'd", "you'd", "we'd", "they'd", "he'd", "she'd",
    "let's",
}
_SUBJECTS = {"i", "we", "they", "he", "she"}
# "be" forms: before an article or a preposition they are the main verb.
_COPULAS = {"am", "is", "are", "was", "were", "be", "been", "being", "i'm", "you're",
            "we're", "they're", "he's", "she's", "it's", "that's", "there's", "here's",
            "what's", "who's"}
# Auxiliaries that are the main verb when an object follows ("have a").
_MAIN_VERB_FORMS = {"have", "has", "had", "having", "do", "does", "did"}
_DETERMINERS = {
    "my", "your", "his", "our", "their", "its", "this", "these", "those",
    "some", "any", "every", "each", "no", "another", "such", "both", "either",
    "neither", "several", "many", "much", "few", "more", "most", "less",
    "least", "other", "same", "whole", "certain",
}
_MODIFIERS = {"not", "very", "really", "quite", "too", "rather", "pretty",
              "extremely", "never", "always", "also", "already", "even",
              "just", "still", "ever", "only", "almost", "nearly"}
NO_LINE_END_EN = (_ARTICLES | _PREPOSITIONS | _CONJUNCTIONS - {"so"} | _AUXILIARIES
                  | _SUBJECTS | _DETERMINERS | {"not", "to", "very", "really",
                                               "quite", "too", "rather"})
SOFT_NO_END_EN = {"her", "never", "always", "also",
                  "already", "even", "just", "still", "ever", "only", "almost",
                  "nearly", "pretty", "extremely", "all", "whole"}
_PARTICLES = {"up", "out", "down", "off", "away", "back", "around", "over",
              "through", "along", "apart", "aside", "together", "forward"}
_OBJECTS = {"it", "them", "him", "us", "me", "her"}
# "you | know", "it | was": these end a line fine as objects, not as subjects.
_MAYBE_SUBJECTS = {"you", "it", "there", "this", "that", "people", "everyone",
                   "everybody", "nobody", "somebody", "someone"}
_COMMON_VERBS = {"know", "think", "want", "see", "need", "like", "mean", "say",
                 "said", "feel", "guess", "believe", "hope", "remember",
                 "understand", "love", "hate", "agree", "seem", "seems",
                 "looks", "sounds", "happen", "happened", "comes", "goes",
                 "gets", "makes", "takes", "works", "means", "wants", "needs"}
_PHRASAL_VERBS = {
    "back", "blow", "break", "bring", "brought", "build", "burn", "call",
    "calm", "carry", "catch", "check", "cheer", "chill", "clean", "clear",
    "close", "come", "came", "count", "cover", "cut", "die", "drag", "dress",
    "drop", "eat", "end", "fall", "fell", "figure", "fill", "find", "found",
    "follow", "freak", "get", "got", "give", "gave", "go", "went", "gone",
    "grow", "grew", "hand", "hang", "hold", "held", "hook", "hurry", "join",
    "keep", "kept", "kick", "knock", "lay", "lead", "led", "leave", "left",
    "let", "lift", "line", "live", "lock", "log", "look", "make", "made",
    "mess", "mix", "move", "open", "pass", "pay", "paid", "pick", "point",
    "pull", "push", "put", "reach", "roll", "run", "ran", "save", "set",
    "settle", "shake", "show", "shut", "sign", "sit", "sat", "slow", "sort",
    "speak", "spoke", "speed", "stand", "stood", "start", "stay", "step",
    "stick", "stop", "sum", "take", "took", "taken", "talk", "tear", "think",
    "thought", "throw", "threw", "tie", "track", "try", "tune", "turn",
    "use", "wait", "wake", "woke", "walk", "warm", "wash", "watch", "wear",
    "wind", "work", "wrap", "write", "wrote", "zip", "grab", "head", "level",
    "line", "max", "plug", "rule", "screw", "shoot", "shot", "sneak", "split",
    "square", "team", "top", "weed", "bring", "dig", "dug", "fight", "fought",
}
_NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "million", "billion", "trillion", "half", "dozen", "couple",
}
_UNITS = {
    "hundred", "thousand", "million", "billion", "trillion", "percent", "%",
    "dollars", "dollar", "pounds", "pound", "euros", "euro", "cents", "cent",
    "years", "year", "months", "month", "weeks", "week", "days", "day",
    "hours", "hour", "minutes", "minute", "seconds", "second", "miles",
    "mile", "kilometres", "kilometers", "kilometre", "kilometer", "meters",
    "metres", "feet", "foot", "inches", "inch", "kilos", "kilograms",
    "grams", "tons", "tonnes", "degrees", "times", "people", "patients",
    "points", "o'clock", "am", "pm", "a.m.", "p.m.",
}
_NUMBER_RE = re.compile(r"^[$£€#]?\d[\d,.:/]*(st|nd|rd|th|s|%|k|m|bn)?$", re.I)
_TITLES = {"mr", "mrs", "ms", "dr", "prof", "st", "sr", "jr", "mt", "rev",
           "gen", "col", "capt", "lt", "sgt", "gov", "sen", "rep", "hon", "pres"}
_TITLE_WORDS = {"president", "senator", "governor", "mayor", "judge", "justice",
                "minister", "secretary", "professor", "doctor", "sir", "madam",
                "lord", "lady", "king", "queen", "prince", "princess", "uncle",
                "aunt", "saint", "officer", "detective", "chief", "director",
                "councilman", "councilwoman", "commissioner", "superintendent",
                "coach", "father", "sister", "brother", "pastor", "reverend"}
_FUNCTION_WORDS = (_ARTICLES | _PREPOSITIONS | _CONJUNCTIONS | _RELATIVES
                   | _AUXILIARIES | _SUBJECTS | _DETERMINERS | _MODIFIERS)


def _bare(token: str) -> str:
    """Lower-case word without surrounding punctuation ("Okay," -> "okay")."""
    t = token.strip().lower().replace("’", "'")
    t = t.strip("\"“”«»()[]{}.,;:!?…—–")
    return t.lstrip("-")


def _verbish(word: str) -> bool:
    if word in _PHRASAL_VERBS:
        return True
    if len(word) > 4 and word.endswith(("ed", "ing")):
        return True
    return word.endswith("s") and word[:-1] in _PHRASAL_VERBS


def _verb_start(word: str) -> bool:
    return word in _AUXILIARIES or word in _COMMON_VERBS or _verbish(word)


def _is_number(word: str) -> bool:
    return bool(_NUMBER_RE.match(word)) or word in _NUMBER_WORDS


def _is_year(word: str) -> bool:
    return bool(re.fullmatch(r"(1[89]|20)\d\d(s)?", word))


# Abbreviations whose period never ends a sentence when a name follows.
_NAME_ABBREVIATIONS = _TITLES | {"messrs", "fr", "cmdr", "supt", "det", "adm", "maj",
                                 "vs", "vol", "fig", "pp", "e.g", "i.e"}


def break_cost(left: Sequence[str], right: Sequence[str], language: str = "en") -> float:
    """Cost of breaking between the words *left* and *right*.

    Negative is a good place to break, positive a bad one. Only the words
    either side of the break (and the one after it) are looked at.
    """
    if not left or not right:
        return 0.0
    a, b = left[-1], right[0]
    a_core = a.rstrip(_CLOSERS) or a
    al, bl = _bare(a), _bare(b)
    english = (language or "en").lower().startswith("en")

    if b.startswith("[") or a.endswith("]"):
        return TAG_EDGE
    if a_core.endswith(_TERMINAL):
        key = al.rstrip(".")
        if a_core.endswith(".") and key in _NAME_ABBREVIATIONS:
            return TITLE_SPLIT if b[:1].isupper() else BAD
        if a_core.endswith(".") and key in NON_TERMINAL_ABBREVIATIONS and not b[:1].isupper():
            return 0.0                                # "the U.S. | government"
        return SENTENCE_END
    if a_core.endswith(_CLAUSE) or (len(a_core) > 1 and a_core.endswith(("-", "--"))):
        return CLAUSE_END

    if not english:
        cost = 0.0
        if al in PREFER_NEW_LINE_WORDS:
            cost += BAD
        if bl in PREFER_NEW_LINE_WORDS:
            cost += GOOD_START
        if bl in CLAUSE_MARKERS["conjunctions"]:
            cost -= 3.0
        return cost

    cost = 0.0
    if al in _MAIN_VERB_FORMS and (bl in _ARTICLES or bl in _DETERMINERS or _is_number(bl)):
        pass                                          # "have | a different ..."
    elif al in _DETERMINERS and (bl in _PREPOSITIONS or bl in _CONJUNCTIONS
                                 or bl in _RELATIVES):
        pass                                          # a pronoun: "after another | through"
    elif al in _COPULAS and (bl in _ARTICLES or bl in _DETERMINERS or bl in _PREPOSITIONS
                             or _is_number(bl)):
        cost += SOFT_BAD                              # "who's | of a certain kind"
    elif al in NO_LINE_END_EN:
        cost += BAD
    elif al in SOFT_NO_END_EN:
        cost += SOFT_BAD
    elif al in _MAYBE_SUBJECTS and _verb_start(bl):
        cost += BAD                                   # "it | was", "you | know"
    if bl in _CONJUNCTIONS or bl in _RELATIVES:
        cost += GOOD_START
    elif bl in _PREPOSITIONS and bl not in _PARTICLES:
        cost += PREP_START
    if bl in _PARTICLES and _verbish(al):
        cost += BAD                                   # phrasal verb
    if bl in _OBJECTS and al not in _CONJUNCTIONS | _RELATIVES | _SUBJECTS:
        cost += OBJECT_SPLIT                          # "enjoy | it", "showing | it"
    if _is_number(al) and (bl in _UNITS or (bl.isalpha() and bl not in _FUNCTION_WORDS
                                            and not _is_year(al))):
        cost += BAD                                   # "21 | patients"
    if (a[:1].isupper() and b[:1].isupper() and al not in _FUNCTION_WORDS
            and bl not in _FUNCTION_WORDS and al != "i" and not al.startswith("i'")):
        cost += BAD                                   # "Henry | Schein"
    elif al in _TITLE_WORDS and b[:1].isupper():
        cost += BAD                                   # "President | Lincoln"
    return cost


def split_cost(text_left: str, text_right: str, language: str = "en") -> float:
    """:func:`break_cost` for two strings."""
    return break_cost(text_left.split(), text_right.split(), language)
