import regex


# Every single character in Unicode has, alongside its ID number, two extra labels attached to it: what category it belongs to (is it a letter? a digit? punctuation? a space?), and what script it belongs to (Latin, Arabic, Cyrillic, Han, and so on). \p{...} is regex's way of asking "match any character that has this label," instead of listing out characters one by one.

ARABIC_SCRIPT=regex.compile(r"\p{Script=Arabic}")
LETTER=regex.compile(r"\p{L}")

all_letters=LETTER.findall("اردو خوبصورت زبان ہے123 iPhone!")
# ['ا', 'ر', 'د', 'و', 'خ', 'و', 'ب', 'ص', 'و', 'ر', 'ت', 'ز', 'ب', 'ا', 'ن', 'ہ', 'ے', 'i', 'P', 'h', 'o', 'n', 'e']


print("All letters in the string:", all_letters)

# \p{Script=Arabic} matches only letters labeled as belonging to the Arabic script (which Urdu uses too, since Urdu is written with an extended version of the Arabic alphabet):


arabic_script=ARABIC_SCRIPT.findall("اردو خوبصورت زبان ہے123 iPhone!")
# ['ا', 'ر', 'د', 'و', 'خ', 'و', 'ب', 'ص', 'و', 'ر', 'ت', 'ز', 'ب', 'ا', 'ن', 'ہ', 'ے']

print(f"Arabic Script is : {arabic_script}")

# function to calculate the script fraction

def script_fraction(text, script_re=ARABIC_SCRIPT):
    letters=LETTER.findall(text)
    if not letters:
        return 0.0
    return len(script_re.findall(text)) / len(letters)


print("Script fraction of Arabic script in the string:", script_fraction("اردو خوبصورت زبان ہے123 iPhone!")) 

print("Script fraction of Arabic script in the string:", script_fraction("Hello, world!")) 

print("Script fraction of Arabic script in the string:", script_fraction("12345!@#$%"))