import contextlib
import re
import signal
from collections import Counter
from importlib.metadata import PackageNotFoundError, version
from math import isclose
from typing import Union


def most_common_element(data):


    assert data and len(data) > 0, "Data is empty"
    return Counter(data).most_common(1)[0][0]


def _check_antlr_version():


    PACKAGE_NAME = 'antlr4-python3-runtime'
    REQUIRED_VERSION = '4.11.0'

    try:
        installed_version = version(PACKAGE_NAME)
        if installed_version != REQUIRED_VERSION:
            raise RuntimeError(
                f"Package {PACKAGE_NAME} version mismatch: {installed_version} (required: {REQUIRED_VERSION})"
            )
    except PackageNotFoundError:
        raise RuntimeError(f"Package {PACKAGE_NAME} not found. Please install antlr4-python3-runtime==4.11.0.")


def _fix_fracs(string):

    while "\\frac " in string:
        string = string.replace("\\frac ", "\\frac")
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
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


def _str_is_int(x: str) -> bool:
    try:
        x = _strip_properly_formatted_commas(x)
        x = float(x)
        return abs(x - int(round(x))) <= 1e-7
    except:
        return False


def _str_to_int(x: str) -> bool:
    x = x.replace(",", "")
    if "_" in x:

        x = x.split("_")[0]
    x = float(x)
    return int(x)


def _inject_implicit_mixed_number(step: str):


    p1 = re.compile("([0-9]) +([0-9])")
    step = p1.sub("\\1+\\2", step)
    return step


def _strip_properly_formatted_commas(expr: str):

    p1 = re.compile(r"(\d)(,)(\d\d\d)($|\D)")
    while True:
        next_expr = p1.sub("\\1\\3\\4", expr)
        if next_expr == expr:
            break
        expr = next_expr
    return next_expr


def _remove_right_units(expr):

    if "\\text" in expr:
        try:
            splits = re.split(r"\\text\s*{\s*", expr)

            assert len(splits) == 2 and splits[0] not in ("", "(")
            return splits[0]
        except AssertionError:
            pass

    if "\\text{" in expr:
        return re.sub(r"\\text{([^}]+)}", r"\1", expr)
    elif "\\mbox{" in expr:
        splits = expr.split("\\mbox{")
        if len(splits) == 2:
            return splits[0]
        else:
            return expr
    else:
        return expr


def _process_and_or_inside_text(string):
    string = re.sub(r"\s*\\text{\s*(or|and)\s*}\s*", ",", string)
    string = re.sub(r",\s*,", ",", string)
    return string


def _remove_left_and_right(expr):

    expr = re.sub(r"\\left", "", expr)
    expr = re.sub(r"\\right", "", expr)
    return expr


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\s*\w+)", r"\\sqrt{\1}", string)
    return _string


def _fix_interval(expr):

    if "\\in " in expr:
        return expr.split("\\in ")[1].strip()

    return expr


def _inject_implicit_mixed_fraction(step: str):


    p1 = re.compile(r"(\d+) *\\frac{(\d+)}{(\d+)}")

    def replacer(match):
        whole_part = match.group(1)
        numerator = match.group(2)
        denominator = match.group(3)

        if whole_part:
            return f"{whole_part} + {numerator}/{denominator}"
        else:
            return f"{numerator}/{denominator}"

    step = p1.sub(replacer, step)
    return step


def normalize_answer_string(expr: str) -> str:

    if expr is None:
        return None


    expr = _remove_left_and_right(expr)
    expr = _process_and_or_inside_text(expr)
    expr = _remove_right_units(expr)
    expr = _fix_interval(expr)
    for surround_str in ["\\\\text", "\\\\mathrm", "\\\\mathcal", "\\\\textbf", "\\\\textit"]:
        expr = expr.replace(surround_str, "")
        pattern = f"^{surround_str}" + r"\{(?P<text>.+?)\}$"
        m = re.search(pattern, expr)
        if m is not None:
            expr = m.group("text")

    expr = expr.replace(r"\!", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\$", "$")
    expr = expr.replace("$", "")
    expr = expr.replace("%", "")
    expr = expr.replace("^{\\circ}", "")

    expr = expr.replace(" or ", " , ")
    expr = expr.replace(" and ", " , ")

    expr = expr.replace("million", "*10^6")
    expr = expr.replace("billion", "*10^9")
    expr = expr.replace("trillion", "*10^12")

    for unit in [
        "degree",
        "cm",
        "centimeter",
        "meter",
        "mile",
        "second",
        "minute",
        "hour",
        "week",
        "month",
        "year",
        "foot",
        "feet",
        "inch",
        "yard",
        "p.m.",
        "PM",
    ]:
        expr = re.sub(rf"{unit}(es)?(s)? *(\^[0-9]+)?", "", expr)

    if "day" in expr:
        days = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        weekday_expressed = False
        for day in days:
            if day in expr:
                weekday_expressed = True
                break

        if not weekday_expressed:
            expr = re.sub(f"day(s)?", "", expr)

    expr = re.sub(rf"\^ *\\\\circ", "", expr)

    if len(expr) > 0 and expr[0] == "{" and expr[-1] == "}":
        expr = expr[1:-1]

    expr = _fix_sqrt(expr)


    expr = _fix_fracs(expr)


    expr = re.sub("- *", "-", expr)
    expr = _inject_implicit_mixed_number(expr)
    expr = _inject_implicit_mixed_fraction(expr)
    expr = expr.replace(" ", "")

    if _str_is_int(expr):
        expr = str(_str_to_int(expr))

    return expr


def is_digit(s):
    try:
        if "{,}" in str(s):
            num = float(str(s).replace("{,}", ""))
            return True, num

        num = float(str(s).replace(",", ""))
        return True, num
    except ValueError:
        return False, None


def normalize(answer) -> str:

    if isinstance(answer, str) and bool(re.match(r'\$\d+(\.\d+)?', answer)):
        return answer[1:]


    if isinstance(answer, str) and (
        bool(re.match(r'^\d+(\.\d+)?%$', answer)) or bool(re.match(r'^\d+(\.\d+)?\\%$', answer))
    ):
        return answer.replace("\\%", "").replace("%", "")

    return answer


def math_equal(
    prediction: Union[bool, float, str],
    reference: Union[float, str],
    include_percentage: bool = True,
    tolerance: float = 1e-4,
    timeout: float = 10.0,
    check_antlr_version: bool = True
) -> bool:


    if check_antlr_version:
        _check_antlr_version()

    from sympy.parsing.sympy_parser import parse_expr

    prediction = normalize(prediction)
    reference = normalize(reference)


    prediction = normalize_answer_string(prediction)
    reference = normalize_answer_string(reference)

    if isinstance(prediction, str) and len(prediction) > 1000:
        prediction = prediction[:1000]


    if isinstance(prediction, str) and isinstance(reference, str):
        if prediction.strip().lower() == reference.strip().lower():
            return True
        if prediction.replace(" ", "") == reference.replace(" ", ""):
            return True

    try:
        if is_digit(prediction)[0] and is_digit(reference)[0]:
            prediction = is_digit(prediction)[1]
            reference = is_digit(reference)[1]

            if include_percentage:
                gt_result = [reference / 100, reference, reference * 100]
            else:
                gt_result = [reference]
            for item in gt_result:
                try:
                    if isclose(item, prediction, rel_tol=tolerance):
                        return True
                except Exception:
                    continue
            return False
    except Exception:
        pass

    if not prediction and prediction not in [0, False]:
        return False


    reference = str(reference).strip()
    prediction = str(prediction).strip()


    prediction = format_intervals(prediction)

    pred_str, ref_str = prediction, reference
    if (prediction.startswith("[") and prediction.endswith("]") and not reference.startswith("(")) or (
        prediction.startswith("(") and prediction.endswith(")") and not reference.startswith("[")
    ):
        pred_str = pred_str.strip("[]()")
        ref_str = ref_str.strip("[]()")
    for s in ["{", "}", "(", ")"]:
        ref_str = ref_str.replace(s, "")
        pred_str = pred_str.replace(s, "")
    if pred_str == ref_str:
        return True


    if (
        prediction
        and reference
        and prediction[0] in "(["
        and prediction[-1] in ")]"
        and prediction[0] == reference[0]
        and prediction[-1] == reference[-1]
    ):
        pred_parts = prediction[1:-1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(pred_pt, ref_pt, include_percentage, tolerance, check_antlr_version=check_antlr_version)
                    for pred_pt, ref_pt in zip(pred_parts, ref_parts)
                ]
            ):
                return True

    if "," in prediction and "," in reference:
        pred_parts = [item.strip() for item in prediction.split(",")]
        ref_parts = [item.strip() for item in reference.split(",")]

        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(pred_parts[i], ref_parts[i], include_percentage, tolerance, check_antlr_version=check_antlr_version)
                    for i in range(len(pred_parts))
                ]
            ):
                return True
            else:
                return False


    if prediction.startswith("Point") and reference[0] == "(" and reference[-1] == ")":
        pred_parts = prediction[prediction.find("(") + 1 : -1].split(",")
        ref_parts = reference[1:-1].split(",")
        if len(pred_parts) == len(ref_parts):
            if all(
                [
                    math_equal(pred_pt, ref_pt, include_percentage, tolerance, check_antlr_version=check_antlr_version)
                    for pred_pt, ref_pt in zip(pred_parts, ref_parts)
                ]
            ):
                return True


    if reference.startswith("\\begin{pmatrix}") and prediction.startswith("Matrix"):
        try:
            pred_matrix = parse_expr(prediction)
            ref_matrix_items = reference.split()[1:-1:2]
            if len(pred_matrix) == len(ref_matrix_items):
                if all(
                    [
                        math_equal(ref, pred, include_percentage, tolerance, check_antlr_version=check_antlr_version)
                        for ref, pred in zip(ref_matrix_items, pred_matrix)
                    ]
                ):
                    return True
        except Exception:
            pass

    return symbolic_equal(prediction, reference, tolerance, timeout)


def symbolic_equal(a, b, tolerance, timeout=10.0):
    import sympy
    from sympy.parsing.latex import parse_latex
    from sympy.parsing.sympy_parser import parse_expr

    def _parse(s):
        for f in [parse_expr, parse_latex]:
            try:
                with time_limit(timeout):
                    return f(s)
            except Exception:
                pass
        return s

    a = _parse(a)
    b = _parse(b)

    try:
        with time_limit(timeout):
            if sympy.simplify(a - b) == 0:
                return True
    except Exception:
        pass

    try:
        with time_limit(timeout):
            if isclose(sympy.N(a), sympy.N(b), rel_tol=tolerance):
                return True
    except Exception:
        pass
    return False


def extract_answer(string: str, extract_from_boxed: bool = True, extract_regex: str = r"The final answer is (.+)$"):

    if not extract_from_boxed:
        match = re.search(extract_regex, string)
        if match:
            return match.group(1)
        return None

    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


class TimeoutException(Exception):
    pass


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def format_intervals(prediction):
    patterns = {
        "Interval(": r"^Interval\((.*)\)$",
        "Interval.Ropen(": r"^Interval\.Ropen\((.*)\)$",
        "Interval.Lopen(": r"^Interval\.Lopen\((.*)\)$",
        "Interval.open(": r"^Interval\.open\((.*)\)$",
    }

    for key, pattern in patterns.items():
        match = re.match(pattern, prediction)
        if match:
            inner_content = match.group(1)

            if key == "Interval(":
                return f"[{inner_content}]"
            elif key == "Interval.Ropen(":
                return f"[{inner_content})"
            elif key == "Interval.Lopen(":
                return f"({inner_content}]"
            elif key == "Interval.open(":
                return f"({inner_content})"

    return prediction

def process_results(
        response: Union[str, list[str]],
        answer: str,
        response_extract_from_boxed: bool = True,
        response_extract_regex: str = r"The final answer is (.+)$",
    ) -> bool:


    if isinstance(response, str):
        return math_equal(
            extract_answer(response, response_extract_from_boxed, response_extract_regex),
            answer,
            check_antlr_version=False,
        )
    elif isinstance(response, list):
        return math_equal(
            most_common_element(
                [
                    extract_answer(r, response_extract_from_boxed, response_extract_regex)
                    for r in response
                ]
            ),
            answer,
            check_antlr_version=False,
        )
    else:
        raise ValueError(f"Invalid response type: {type(response)}")

def reward_func(data_source, solution_str, ground_truth) -> float:
    extracted_answer = extract_answer(solution_str, extract_from_boxed=True)
    if extracted_answer is None:
        return -1.0
    else:
        if math_equal(extracted_answer, ground_truth, check_antlr_version=False):
            return 1.0
        else:
            return -0.5
