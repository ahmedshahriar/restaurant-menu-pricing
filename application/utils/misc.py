import html

import pandas as pd


def unescape_html(value):
    """Unescape HTML entities in a string if the value is not null."""
    if pd.notnull(value):
        return html.unescape(value)
    return value


def convert_entities_to_list(text, entities: list[dict]) -> list[str]:
    """Convert a list of entity dicts to a list of entity strings, merging adjacent entities of the same type."""
    ents = []
    for ent in entities:
        e = {"start": ent["start"], "end": ent["end"], "label": ent["entity_group"]}
        if ents and -1 <= ent["start"] - ents[-1]["end"] <= 1 and ents[-1]["label"] == e["label"]:
            ents[-1]["end"] = e["end"]
            continue
        ents.append(e)

    return [text[e["start"] : e["end"]] for e in ents]
