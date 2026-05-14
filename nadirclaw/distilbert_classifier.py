"""DistilBERT-based complexity classifier.

Fine-tunes distilbert-base-uncased on the same training prototypes as the
TrainedClassifier, but uses DistilBERT's native sequence classification head
instead of sentence embeddings + GBM.

Comparison baseline for the trained classifier.
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_MODEL_DIR = os.path.join(_PKG_DIR, "distilbert_model")
# Published artifact (~256MB). from_pretrained() downloads + caches it under
# ~/.cache/huggingface/hub on first use. Override with NADIRCLAW_DISTILBERT_REPO.
_HF_REPO = os.getenv("NADIRCLAW_DISTILBERT_REPO", "amirdor/nadirclaw-distilbert")
_TIER_MAP = {0: "simple", 1: "medium", 2: "complex"}


class DistilBertClassifier:
    """DistilBERT sequence classifier for prompt complexity (3-class)."""

    CLASSIFIER_VERSION = "1.0"

    def __init__(self):
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

        self._device = "cpu"  # keep on CPU for consistency with other classifiers

        # Resolution order:
        #   1. Local fine-tuned dir (dev machines, offline use)
        #   2. Hugging Face Hub repo (downloaded + cached on first use)
        #   3. Train from scratch on local prototypes (last resort)
        if os.path.isdir(_MODEL_DIR):
            logger.info("Loading fine-tuned DistilBERT from %s", _MODEL_DIR)
            self._tokenizer = DistilBertTokenizer.from_pretrained(_MODEL_DIR)
            self._model = DistilBertForSequenceClassification.from_pretrained(_MODEL_DIR)
        else:
            try:
                logger.info("Loading fine-tuned DistilBERT from HF Hub: %s", _HF_REPO)
                self._tokenizer = DistilBertTokenizer.from_pretrained(_HF_REPO)
                self._model = DistilBertForSequenceClassification.from_pretrained(_HF_REPO)
            except Exception as e:
                logger.warning(
                    "Could not load DistilBERT from HF Hub (%s) — training from scratch",
                    e,
                )
                self._tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
                self._model = DistilBertForSequenceClassification.from_pretrained(
                    "distilbert-base-uncased", num_labels=3
                )
                self._train_on_prototypes()
                self._save_model()

        self._model.to(self._device)
        self._model.eval()
        logger.info("DistilBertClassifier ready (device=%s)", self._device)

    def _get_training_data(self) -> Tuple[List[str], List[int]]:
        """Load training prototypes — same data as TrainedClassifier."""
        from nadirclaw.prototypes import (
            SIMPLE_PROTOTYPES,
            MEDIUM_PROTOTYPES,
            COMPLEX_PROTOTYPES,
        )

        texts = []
        labels = []

        for p in SIMPLE_PROTOTYPES:
            texts.append(p if isinstance(p, str) else p.get("text", str(p)))
            labels.append(0)
        for p in MEDIUM_PROTOTYPES:
            texts.append(p if isinstance(p, str) else p.get("text", str(p)))
            labels.append(1)
        for p in COMPLEX_PROTOTYPES:
            texts.append(p if isinstance(p, str) else p.get("text", str(p)))
            labels.append(2)

        # Try to load external training data (Horizen prototypes, eval prompts)
        try:
            from nadirclaw.trained_classifier import _load_external_training_data
            ext_texts, ext_labels = _load_external_training_data()
            texts.extend(ext_texts)
            labels.extend(ext_labels)
        except (ImportError, Exception) as e:
            logger.debug("No external training data: %s", e)

        logger.info("Training data: %d samples (%d simple, %d medium, %d complex)",
                     len(texts), labels.count(0), labels.count(1), labels.count(2))
        return texts, labels

    def _train_on_prototypes(self):
        """Fine-tune DistilBERT on prototype data."""
        from torch.utils.data import DataLoader, TensorDataset

        texts, labels = self._get_training_data()

        # Tokenize
        encodings = self._tokenizer(
            texts, truncation=True, padding=True, max_length=256, return_tensors="pt"
        )
        dataset = TensorDataset(
            encodings["input_ids"],
            encodings["attention_mask"],
            torch.tensor(labels, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Fine-tune
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=2e-5, weight_decay=0.01)
        self._model.train()

        n_epochs = 10
        for epoch in range(n_epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids, attention_mask, batch_labels = batch
                input_ids = input_ids.to(self._device)
                attention_mask = attention_mask.to(self._device)
                batch_labels = batch_labels.to(self._device)

                outputs = self._model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=batch_labels,
                )
                loss = outputs.loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 2 == 0:
                logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, n_epochs, total_loss / len(loader))

        self._model.eval()
        logger.info("DistilBERT fine-tuning complete")

    def _save_model(self):
        """Save fine-tuned model to disk."""
        os.makedirs(_MODEL_DIR, exist_ok=True)
        self._model.save_pretrained(_MODEL_DIR)
        self._tokenizer.save_pretrained(_MODEL_DIR)
        logger.info("Saved fine-tuned DistilBERT to %s", _MODEL_DIR)

    def classify(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        """Classify a prompt into simple/medium/complex."""
        start = time.time()

        inputs = self._tokenizer(
            prompt, truncation=True, padding=True, max_length=256, return_tensors="pt"
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            logits = outputs.logits[0]
            probs_tensor = torch.softmax(logits, dim=-1)
            probs = probs_tensor.cpu().numpy()

        pred_idx = int(np.argmax(probs))
        tier = _TIER_MAP[pred_idx]
        confidence = float(probs[pred_idx])

        # Same safety escalation as TrainedClassifier
        escalated = False
        if tier == "simple" and confidence < 0.70:
            if probs[2] >= probs[1]:
                tier, confidence = "complex", float(probs[2])
            else:
                tier, confidence = "medium", float(probs[1])
            escalated = True

        classify_ms = int((time.time() - start) * 1000)

        metadata = {
            "tier_probabilities": {
                "simple": float(probs[0]),
                "medium": float(probs[1]),
                "complex": float(probs[2]),
            },
            "confidence_escalated": escalated,
            "stage1_tier": _TIER_MAP[pred_idx],
            "stage1_confidence": float(probs[pred_idx]),
            "classify_ms": classify_ms,
            "classifier_version": self.CLASSIFIER_VERSION,
        }

        return tier, confidence, metadata

    async def analyze(self, text: str = "", system_message: str = "", **kwargs) -> Dict[str, Any]:
        """Async-compatible interface matching the server's expected API."""
        tier, confidence, meta = self.classify(text)

        probs = meta.get("tier_probabilities", {})
        p_s = probs.get("simple", 0)
        p_m = probs.get("medium", 0)
        p_c = probs.get("complex", 0)

        # Same calibrated score formula as TrainedClassifier (lean complex)
        if tier == "simple":
            complexity_score = (1.0 - p_s) * 0.40
        elif tier == "complex":
            complexity_score = 1.0 - (1.0 - p_c) * 0.40
        else:
            # Medium: [0.40, 0.75] — lean complex. Use probability ratio
            # when available, prompt length heuristic when GBM is confident-medium
            denom = p_s + p_c
            if denom > 0.05:
                ratio = p_c / denom
            else:
                prompt_len = len(text)
                ratio = min(1.0, max(0.0, (prompt_len - 80) / 400))
            complexity_score = 0.40 + ratio * 0.35

        # The rest of NadirClaw routes on tier_name == "mid" (not "medium").
        # Normalize so the server can dispatch to MID_MODEL without translation.
        normalized_tier = "mid" if tier == "medium" else tier

        return {
            "tier_name": normalized_tier,
            "confidence": confidence,
            "complexity_score": complexity_score,
            "analyzer_type": f"distilbert-v{self.CLASSIFIER_VERSION}",
            "analyzer_latency_ms": meta.get("classify_ms", 0),
            "reasoning": f"DistilBertClassifier: {normalized_tier} ({confidence:.0%})",
            "ranked_models": [],
            **meta,
        }
