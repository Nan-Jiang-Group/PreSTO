"""
This logic is largely copied from the Hendrycks' MATH release (math_equivalence).
"""
import re


def normalize_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    try:
        # Remove enclosing `\text{}`.
        m = re.search("^\\\\text\\{(?P<text>.+?)\\}$", answer)
        if m is not None:
            answer = m.group("text").strip()
        return _strip_string(answer)
    except:
        return answer


def _strip_string(string):
    # Apply all replacements
    replacements = [
        ("\n", ""),  # linebreaks
        ("\\!", ""),  # remove inverse spaces
        ("\\\\", "\\"),  # replace \\ with \
        ("tfrac", "frac"),  # replace tfrac with frac
        ("dfrac", "frac"),  # replace dfrac with frac
        ("\\left", ""),  # remove \left
        ("\\right", ""),  # remove \right
        ("^{\\circ}", ""),  # remove circ (degrees)
        ("^\\circ", ""),  # remove circ (degrees)
        ("\\$", ""),  # remove dollar signs
        ("\\%", ""),  # remove percentage
        ("%", ""),  # remove bare percentage
        (" .", " 0."),  # " ." equivalent to " 0."
        ("{.", "{0."),  # "{." equivalent to "{0."
    ]

    for old, new in replacements:
        string = string.replace(old, new)

    # Remove units (on the right)
    string = _remove_right_units(string)

    # Add "0" if "." is the start of the string
    if string and string[0] == ".":
        string = "0" + string

    # Remove simple variable assignments (e.g., "k = " or "q = " at beginning)
    parts = string.split("=")
    if len(parts) == 2 and len(parts[0]) <= 2:
        string = parts[1]

    # Fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # Remove spaces
    string = string.replace(" ", "")

    # Fix fractions: \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}
    string = _fix_fracs(string)

    # Manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # Fix X/Y changed to \frac{X}{Y}
    string = _fix_a_slash_b(string)

    return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string
