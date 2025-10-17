import math

import numpy as np
import pandas as pd


def test_unescape_html_handles_amp_and_nbsp_tolerantly():
    from application.utils.misc import unescape_html

    # &amp; should be unescaped
    assert unescape_html("Tom &amp; Jerry") == "Tom & Jerry"

    # Be tolerant: &nbsp; may decode to NBSP (\u00A0) or a normal space later
    out = unescape_html("Hello&nbsp;World")
    assert out == "Hello\u00a0World" or out.replace("\u00a0", " ") == "Hello World"


def test_unescape_html_preserves_null_like_values():
    from application.utils.misc import unescape_html

    assert unescape_html(None) is None
    v = unescape_html(np.nan)
    assert isinstance(v, float) and math.isnan(v)
    v2 = unescape_html(pd.NA)
    assert pd.isna(v2)


def test_convert_entities_to_list_merges_adjacent_and_overlapping_same_label():
    from application.utils.misc import convert_entities_to_list

    text = "tomato basil pesto"
    ents = [
        {"start": 0, "end": 6, "entity_group": "ING"},  # "tomato"
        {"start": 7, "end": 12, "entity_group": "ING"},  # "basil" (gap=1) -> merge
        {"start": 11, "end": 18, "entity_group": "ING"},  # overlaps into "pesto" -> merge
    ]
    result = convert_entities_to_list(text, ents)
    assert result == ["tomato basil pesto"]


def test_convert_entities_to_list_keeps_separate_when_labels_differ():
    from application.utils.misc import convert_entities_to_list

    text = "spicy tomato soup"
    ents = [
        {"start": 0, "end": 5, "entity_group": "LAB"},  # "spicy"
        {"start": 6, "end": 12, "entity_group": "ING"},  # "tomato"
    ]
    result = convert_entities_to_list(text, ents)
    assert result == ["spicy", "tomato"]


def test_convert_entities_to_list_single_and_empty_inputs():
    from application.utils.misc import convert_entities_to_list

    text = "avocado toast"
    assert convert_entities_to_list(text, [{"start": 0, "end": 7, "entity_group": "ING"}]) == ["avocado"]
    assert convert_entities_to_list(text, []) == []


def test_convert_entities_to_list_boundary_gap_character_kept():
    from application.utils.misc import convert_entities_to_list

    text = "red-onion"
    ents = [
        {"start": 0, "end": 3, "entity_group": "ING"},  # "red"
        {"start": 4, "end": 9, "entity_group": "ING"},  # "onion" (gap is "-")
    ]
    out = convert_entities_to_list(text, ents)
    assert out == ["red-onion"]
