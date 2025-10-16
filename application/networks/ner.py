import torch
from loguru import logger
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from core.settings import settings

from .base import SingletonMeta


# === Food Ingredient Extraction via RoBERTa NER Model ===
class NERModelSingleton(metaclass=SingletonMeta):
    """
    Singleton class to manage a HuggingFace NER pipeline instance.
    Loads the model and tokenizer once, and provides access to the pipeline.
    1. Thread-safe singleton implementation.
    2. Automatically selects device (CUDA, MPS, CPU).
    3. Loads model specified in settings.NER_MODEL.
    4. Provides get_pipeline() method to access the NER pipeline.
    Usage:
        ner_instance = NERModelSingleton()
        ner_pipeline = ner_instance.get_pipeline()
    5. Logs loading status and errors.
    """

    def __init__(self):
        model_name = settings.NER_MODEL

        # --- Select device: CUDA → MPS → CPU ---
        device_str = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

        logger.info(f"Loading NER model '{model_name}' on device='{device_str}' ...")

        try:
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            model = AutoModelForTokenClassification.from_pretrained(model_name)

            # Move model to the right device
            model = model.to(device_str)

            # HuggingFace pipeline — let it manage only CPU/GPU indexing
            self.pipeline = pipeline(  # type: ignore
                task="ner",
                model=model,
                tokenizer=tokenizer,
            )

        except Exception as e:
            logger.exception(f"Failed to load NER model '{model_name}': {e}")
            raise RuntimeError(f"Failed to initialize NER model '{model_name}'") from e

        logger.info(f"NER pipeline loaded successfully on {device_str}.")

    def get_pipeline(self):
        """Return the singleton NER pipeline instance."""
        return self.pipeline
