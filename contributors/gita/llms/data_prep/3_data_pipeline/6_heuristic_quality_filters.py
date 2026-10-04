import re

STOP_WORDS={"the", "be", "to", "of", "and", "that", "have", "with"}


def gopher_quality(text, min_words=50, max_words=100_0000):
    """Simplified gopher rules, returns (passed, reason)"""

    words=text.split()

    n=len(words)
    if not (min_words<=n<=max_words):
        return False, f"word count {n} outside [{min_words}, {max_words}]"
    mean_len=sum(len(w) for w in words)/n

    if not (3<=mean_len<=20):
        return False, f"mean word length {mean_len:.1f} outside [3, 20]"    
    if text.count("#")/n >0.1:
        # hash ratio
        return False, f"too many hashtags {text.count('#')/n:.2%} > 10%"
    if text.count("...")/n > 0.1:
        return False, "ellipsis_ratio"
    lines=[l for l in text.split("\n") if l.strip()]

    if lines:
        bullet=sum(l.lstrip().startswith(("\u2022", "-", "*")) for l in lines)
        if bullet > 0.9:
            return False, "bullet_lines"
        ellip=sum(l.rstrip().endswith(("...", "\u2026")) for l in lines)/len(lines)
        if ellip > 0.3:
            return False, "ellipsis_lines"
    alpha=sum(any(ch.isalpha() for ch in w) for w in words)/n
    if alpha < 0.8:
        return False, "alpha_words"
    if len(STOP_WORDS & {w.lower() for w in words}) < 2:
        return False, "stop_words"
    return True, "passed"


testing_clean=gopher_quality("This is a test text with enough words to pass the filter. It should be long enough and contain some stop words like 'the' and 'and'. Let's see if it passes the quality check.", min_words=10, max_words=100)

print("____________________the clean result here_______________")
print(f"RESULTS: {testing_clean}")