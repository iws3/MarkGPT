# Module 01 — Reading Notes

Read as a Day 1 beginner would, following the README's own instruction to
"Start with GETTING_STARTED.md, then work through Module 01 lesson by lesson."

## 1. Module 01's content belongs to Module 05

`modules/module-01/README.md` covers NLP foundations, tokenization, text
preprocessing, linguistic concepts, and NLTK/spaCy.

But `SYLLABUS.md` line 6 defines Module 01 as *"Foundations: What Is AI? What Is
Language?"*, Days 1–6 — the history of AI, what a language model is, environment
setup, and the BansoGram mini-project. The root README's phase table agrees.
The content currently in module-01 matches the syllabus's **Module 05 —
NLP Foundations, Text as Data** (Days 25–30).

It looks like module content was placed one-to-one against the wrong module
numbers. Worth checking whether modules 02–10 are shifted the same way.

## 2. This breaks the "no prerequisites" promise

The README states prerequisites are "None. Truly. If you can read this sentence,
you can begin Day 1," and Python is taught in Module 02 (Days 7–12). As it
stands, a Day 1 learner meets NLTK and spaCy before the course has introduced a
variable, a function, or `pip install`. The syllabus's actual Module 01 avoids
this correctly — it opens with reading and a Hugging Face Spaces demo, no local
Python required.

## 3. Module 01 has headings but no lessons

The file lists five section titles with a one-line gloss each. The syllabus
specifies named lessons per day (`L01.1`–`L01.4`), exercises (`E01.1`–`E01.3`),
and cited resources. None of that is present, so a student following "lesson by
lesson" has no lessons to follow.

## 4. Contradiction: where does student work go?

`SYLLABUS.md:129` tells students to submit the Day 6 project to
`modules/module-01/projects/banso_gram.py`.

`contributors/CONTRIBUTORS_GUIDE.md`, Don'ts #2, says: *"Don't Modify Files
Outside Your Contributor Folder — only edit files in contributors/YOUR-USERNAME/."*

A student following the syllabus violates the contributors guide. Worth deciding
which is authoritative and making the other match.

## Suggestion

The renumbering (1) is the one to settle first, since everything else depends on
which content lands where. Happy to draft the Module 01 lessons against the
syllabus spec if that would help.