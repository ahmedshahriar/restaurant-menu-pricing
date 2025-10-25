from __future__ import annotations

from dataclasses import dataclass

from core import settings


@dataclass
class Config:
    # Kaggle dataset
    RESTAURANTS_DS: str = "ahmedshahriarsakib/uber-eats-usa-restaurants-menus/versions/12"
    RESTAURANTS_FILE: str = "restaurants.csv"

    MENUS_DS: str = "ahmedshahriarsakib/uber-eats-usa-restaurants-menus/versions/12"
    MENUS_FILE: str = "restaurant-menus.csv"

    # TODO: update with latest data
    INDEX_DS: str = settings.INDEX_DS
    INDEX_FILE: str = settings.INDEX_FILE

    DENSITY_DS: str = settings.DENSITY_DS
    DENSITY_FILE: str = settings.DENSITY_FILE

    # used to build states_name_dict
    STATES_DS: str = settings.STATES_DS
    STATES_FILE: str = settings.STATES_FILE

    # TODO: add dynamic config to select categories
    # Category sampling choices
    focus_categories: tuple[str, ...] = ("Sandwiches", "Salads", "Wraps")  # top 3 categories by count
    top_categories_per_city: int = 15  # constraint
    top_cities_per_state: int = 5  # constraint

    # TODO: add dynamic config to select states
    # State filtering (to focus on top states by count)
    top_states_filter: tuple[str, ...] = ("tx", "va", "wa", "wi", "ut")  # top 5 states by count

    # NER model
    NER_MODEL: str = settings.NER_MODEL

    # Output sampled final featured data
    FINAL_SAMPLED_DATA_PATH: str = settings.SAMPLED_DATA_PATH
