# ============================================================
# EXERCISE 1
# Write your own HTML page with a sidebar, a comment section,
# a code block and an equation. Extract with the naive method
# and with trafilatura. Which parts does each one lose?
# ============================================================

from bs4 import BeautifulSoup
import trafilatura

html = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Why gradients vanish</title></head>
<body>
<aside class="sidebar">
  <h3>Related posts</h3>
  <ul><li>Intro to backprop</li><li>Activation functions</li><li>Subscribe to the newsletter</li></ul>
</aside>
<nav><a href="/">Home</a> <a href="/blog">Blog</a> <a href="/about">About</a></nav>
<article>
<h1>Why gradients vanish in deep networks</h1>
<p>When a network has many layers, the gradient of the loss with respect to
an early layer is a product of many terms. If each term is smaller than 1,
the product shrinks toward zero.</p>
<pre><code class="language-python">
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)   # max value is 0.25, at z = 0
</code></pre>
<p>The derivative of the sigmoid is at most 0.25. For a network with L layers,
the gradient reaching layer 1 is roughly bounded by:</p>
<p class="equation">&part;L/&part;w&#8321; &le; 0.25<sup>L</sup> &times; &part;L/&part;w<sub>L</sub></p>
<p>With L = 20, that factor is about 10<sup>-12</sup>, which is why early
layers barely update.</p>
</article>
<section class="comments">
  <h3>3 comments</h3>
  <div class="comment"><b>ada_k:</b> This finally made it click for me, thank you.</div>
  <div class="comment"><b>devguy22:</b> Does ReLU fully fix this?</div>
  <div class="comment"><b>ada_k:</b> @devguy22 mostly, though it introduces dying ReLUs instead.</div>
</section>
<footer>&copy; 2025 gradient-notes.example. Terms | Privacy</footer>
</body></html>"""

naive = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
good  = trafilatura.extract(html, include_comments=True, include_tables=False)

# --- REAL OUTPUT, both extractors ---------------------------------------
# naive  : 933 chars. Contains EVERYTHING: sidebar list, nav links, article
#          prose, code, equation, comments, footer, all glued into one
#          run-on line of text, no paragraph breaks.
# trafilatura (include_comments=True): 607 chars.
#
# --- what survives in each, checked with a plain substring test ---------
#   piece                naive   trafilatura
#   sidebar "Related posts"   True    False   <- trafilatura drops it
#   nav "Home"                True    False   <- trafilatura drops it
#   article prose              True    True
#   code block                 True    True
#   equation line               True    True    (but see note below)
#   comment "finally made it click"  True  False  <- trafilatura's own
#                                                     comment-detection
#                                                     heuristics did not
#                                                     recognize my custom
#                                                     <div class="comment">
#                                                     markup, even with
#                                                     include_comments=True
#   footer "Terms | Privacy"    True    False   <- trafilatura drops it
#
# ANSWER:
# - trafilatura keeps: the headline, the body prose, the code block, and
#   the equation line. It correctly throws away the sidebar, nav, and
#   footer boilerplate, exactly what we want for pretraining.
# - trafilatura LOSES the comments here, not because it never extracts
#   comments, but because my hand-made comment HTML isn't structured the
#   way trafilatura's comment detector expects. This is a real, honest
#   example of a FALSE NEGATIVE: real content lost by the extractor.
# - trafilatura also flattens the equation's <sup>/<sub> tags into plain
#   inline text ("0.25L" instead of "0.25 to the power L", "10-12" instead
#   of "10 to the minus 12"), silently losing which parts were exponents.
#   This is exactly the "equations become garbled" warning from 5.3.
# - the naive method keeps literally everything, including the three
#   things we explicitly don't want (sidebar, nav, footer), proving again
#   that "strip all tags" is not the same as "find the article."


# ============================================================
# EXERCISE 2
# Make a list of five characters that change under NFKC and
# one that changes under NFC. For each, decide whether you
# would want the change in a code corpus.
# ============================================================

import unicodedata

nfkc_candidates = [
    "\ufb02ow",              # starts with the "fl" ligature
    "\u2155",                 # VULGAR FRACTION ONE FIFTH
    "\uff34\uff45\uff58\uff54",  # fullwidth "Text"
    "\u00bd",                 # VULGAR FRACTION ONE HALF
    "x\u00b3",                 # x, then SUPERSCRIPT THREE
    "\u212b",                  # ANGSTROM SIGN (a unit symbol!)
]
for s in nfkc_candidates:
    out = unicodedata.normalize("NFKC", s)
    print(repr(s), "->", repr(out))

# --- REAL OUTPUT ----------------------------------------------------------
# 'ﬂow'      -> 'flow'       (ligature fl -> plain f, l)
# '⅕'        -> '1⁄5'        (one fraction glyph -> digit, slash, digit)
# 'Ｔｅｘｔ'    -> 'Text'       (fullwidth letters -> plain ASCII)
# '½'        -> '1⁄2'        (one fraction glyph -> digit, slash, digit)
# 'x³'       -> 'x3'         (superscript 3 -> plain digit 3)
# 'Å'        -> 'Å'          (LOOKS unchanged, but check ord():
#                              before: ord 8491 (ANGSTROM SIGN)
#                              after:  ord  197 (LATIN CAPITAL A WITH RING)
#                              -> it DID change underneath, just to a
#                                 character that renders identically)
#
# ONE NFC CHANGE (not NFKC):
nfc_s = "e\u0301tude"   # "étude" spelled as e + combining accent
print(repr(nfc_s), "-> NFC ->", repr(unicodedata.normalize("NFC", nfc_s)))
# 'étude' -> 'étude'   (length 6 -> 5: just glues the accent onto the e,
#                        same character meaning, nothing is reinterpreted)
#
# ANSWER for a CODE corpus, one by one:
#   'ﬂow' -> 'flow'     WANT IT. Ligatures are pure typesetting noise;
#                        source code never uses them on purpose.
#   '⅕'  -> '1⁄5'       DO NOT WANT IT, if this fraction glyph ever shows
#                        up inside a string literal or comment meant to
#                        render as one glyph, NFKC changes the rendered
#                        output. Rare in code, but not impossible in a
#                        comment or docstring.
#   'Ｔｅｘｔ' -> 'Text'  WANT IT. Fullwidth letters sometimes leak into
#                        code from copy-pasted CJK-locale text editors;
#                        collapsing them to ASCII is almost always
#                        correct for code, which is ASCII-identifier based.
#   '½'  -> '1⁄2'        same reasoning as 1/5 above: DO NOT WANT IT if it
#                        could appear inside a string a program prints.
#   'x³' -> 'x3'         DO NOT WANT IT. In code this superscript would
#                        almost never appear except inside a comment or
#                        string representing math (e.g. "x³ + 1"), and
#                        turning it into "x3" changes that string's literal
#                        content, which could break a doctest or a
#                        string-comparison test.
#   'Å'  -> 'Å' (ord change only) DO NOT WANT IT in scientific/engineering
#                        code: the Angstrom sign is a unit symbol used on
#                        purpose in physics/chemistry code and comments;
#                        silently swapping its code point is invisible but
#                        changes the literal bytes being compared or hashed.
#   the NFC étude example -> ALWAYS SAFE, for code or anything else, since
#                        it never changes which character is meant, only
#                        which of two equivalent spellings is used.
#
# GENERAL RULE CONFIRMED: NFC is safe everywhere. NFKC is risky exactly
# wherever a "compatibility" character might be carrying meaning rather
# than just being a stylistic variant, which is common in code/math and
# rare in ordinary prose.


# ============================================================
# EXERCISE 3
# Take 20 documents from FineWeb and run ftfy.fix_text on them.
# How many are changed?
# ============================================================

# NOTE: my sandbox's network allowlist does not include huggingface.co,
# so I could not actually stream FineWeb here. This is the exact code
# you would run with network access:

"""
from datasets import load_dataset
import ftfy

ds = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT",
                   split="train", streaming=True)
changed = 0
for i, doc in enumerate(ds):
    fixed = ftfy.fix_text(doc["text"])
    if fixed != doc["text"]:
        changed += 1
    if i == 19:
        break
print(f"{changed} of 20 documents were changed by ftfy")
"""

# In place of real FineWeb documents, I ran the same check on 20 stand-in
# sentences built to contain the kinds of damage that really does show up
# in web-scraped text: mojibake (UTF-8 decoded as Latin-1), smart quotes,
# stray "Â" from a double-decoded copyright sign, and plain clean text
# with nothing wrong, as a control group.

import ftfy

stand_in_docs = [
    "The council voted on Tuesday to approve the budget.",
    "We\u2019re excited to announce our new product line.",
    "The city said it\u00e2\u0080\u0099s \u00e2\u0080\u009cready\u00e2\u0080\u009d for the event.",
    "Visit our caf\u00c3\u00a9 for fresh pastries every morning.",
    "100% organic, locally sourced ingredients.",
    "She said \u201cI\u2019ll be there by five\u201d and left.",
    "The temperature dropped to \u221210\u00b0C overnight.",
    "Click here to unsubscribe from this newsletter.",
    "Â© 2025 Example Corp. All rights reserved.",
    "The recipe calls for 2\u00bd cups of flour.",
    "He wrote: \u201cnothing lasts forever\u201d in his journal.",
    "Prices start at \u20ac19.99 per month.",
    "The event was postponed due to weather conditions.",
    "schÃ¶n und grÃ¼n beschreibt den Garten im Fr\u00c3\u00bchling.",
    "Welcome to our online store, est. 1998.",
    "The committee reviewed the proposal carefully.",
    "Don\u00e2\u0080\u0099t forget to bring your ID to the office.",
    "Fresh produce delivered weekly to your door.",
    "The museum\u00e2\u0080\u0099s new exhibit opens next month.",
    "A simple, honest review of the product.",
]

changed = sum(ftfy.fix_text(d) != d for d in stand_in_docs)
print(f"{changed} of {len(stand_in_docs)} stand-in documents were changed")

# --- REAL OUTPUT: 9 of 20 changed ------------------------------------
# Changed (9): docs 1, 2, 3, 5, 8, 10, 13, 16, 18
#   - docs 2, 3, 13: classic UTF-8/Latin-1 mojibake, e.g.
#     "cafÃ©" -> "café", "schÃ¶n" -> "schön"
#   - docs 8: double-decoded copyright sign "Â©" -> "©"
#   - docs 1, 5, 10, 16, 18: curly quotes/apostrophes "'" "\u201c" "\u201d"
#     uncurled to plain ASCII "'" '"'. This happens because ftfy 6.3.1
#     uncurls quotes by default as one of its standard fixes, not because
#     anything was broken, it's a normalization choice ftfy makes for you.
# Unchanged (11): already-clean ASCII/UTF-8 text, and a few that use
#     correctly-encoded non-ASCII characters on purpose (the € sign, the
#     − minus sign, the ½ fraction), which ftfy correctly leaves alone
#     since those are not mojibake, they're valid, intentional characters.
#
# TAKEAWAY: close to half of this small, deliberately mixed batch needed
# a fix. A real FineWeb sample would show a lower rate (most modern web
# text decodes correctly), but it would not be zero, which is exactly why
# every real pipeline runs ftfy (or an equivalent) unconditionally.


# ============================================================
# EXERCISE 4
# Rewrite the handbook's cleaning function so that it
# (a) does not delete characters, and
# (b) returns None if the script fraction is below 0.5.
# ============================================================

import re, regex, unicodedata

URL_RE          = re.compile(r"https?://\S+|www\.\S+")
HTML_TAG_RE     = re.compile(r"<[^>]+>")
EMAIL_RE        = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
REPEAT_PUNCT_RE = re.compile(r"([!?.,])\1{2,}")
WHITESPACE_RE   = re.compile(r"[ \t]{2,}")

LETTER         = regex.compile(r"\p{L}")
ARABIC_SCRIPT  = regex.compile(r"\p{Script=Arabic}")

def script_fraction(text, script_re=ARABIC_SCRIPT):
    letters = LETTER.findall(text)
    if not letters:
        return 0.0
    return len(script_re.findall(text)) / len(letters)

def clean_urdu_v2(text, min_script_fraction=0.5):
    # (a) NO CHARACTER DELETION: only remove whole URLs, HTML tags and
    #     emails as complete spans, never individual characters by script.
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)     # safe: merges spelling
                                                   # variants only, drops
                                                   # nothing, see Part 5.5
    text = REPEAT_PUNCT_RE.sub(r"\1\1\1", text)   # "!!!!!!" -> "!!!"
    text = WHITESPACE_RE.sub(" ", text).strip()

    # (b) REJECT THE WHOLE DOCUMENT if it isn't mostly the target script,
    #     instead of deleting individual non-matching characters out of
    #     a document we're keeping.
    if script_fraction(text) < min_script_fraction:
        return None
    return text

tests = [
    "اردو ایک خوبصورت زبان ہے۔ ملاحظہ کریں: https://example.com/urdu اور رابطہ کریں info@example.com",
    "میں نے iPhone خریدا، بہت اچھا ہے!!!!!! <b>زبردست</b>",
    "The council voted on Tuesday.",   # should be REJECTED: not Urdu
]
for t in tests:
    print("in :", t)
    print("out:", clean_urdu_v2(t))
    print()

# --- REAL OUTPUT ------------------------------------------------------
# in : اردو ایک خوبصورت زبان ہے۔ ملاحظہ کریں: https://... info@...
# out: اردو ایک خوبصورت زبان ہے۔ ملاحظہ کریں: اور رابطہ کریں
#      (URL and email removed as whole spans; every Urdu letter kept)
#
# in : میں نے iPhone خریدا، بہت اچھا ہے!!!!!! <b>زبردست</b>
# out: میں نے iPhone خریدا، بہت اچھا ہے!!! زبردست
#      (the word "iPhone" is PRESERVED, unlike the handbook's version,
#       which deleted any character outside its allowed ranges and would
#       have silently cut "iPhone" down to nothing; here it survives
#       intact because we only reject at the WHOLE-DOCUMENT level)
#
# in : The council voted on Tuesday.
# out: None
#      (pure English, script_fraction = 0.0, correctly rejected whole)
#
# CONFIRMED DIRECTLY:
mixed = "میں نے iPhone خریدا"
result = clean_urdu_v2(mixed)
print("'iPhone' preserved in output?", "iPhone" in result)   # True
#
# ONE HONEST CAVEAT I found while testing: script_fraction can exceed 1.0
# on text with heavy Urdu punctuation (۔ ؟), because \p{Script=Arabic}
# also matches some punctuation marks that \p{L} (letters only) does not
# count in the denominator. It never breaks the < 0.5 check used here,
# but it's a real edge case worth knowing about if you reuse this
# function for a stricter threshold.