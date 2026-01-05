"""Semantic search for HED tags using embeddings and keyword matching.

This module provides semantic search capabilities to find relevant HED tags
based on natural language queries. It uses a dual approach:
1. Deterministic keyword index for exact/known term matches
2. Embedding-based search for semantic similarity

Reference: Ported from hed-lsp (https://github.com/hed-standard/hed-lsp)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Default embedding model
DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"


@dataclass
class TagMatch:
    """A matched HED tag with relevance score."""

    tag: str
    long_form: str
    prefix: str  # "" for base schema, "sc:" for SCORE, etc.
    score: float
    source: Literal["keyword", "embedding", "both"]

    def __repr__(self) -> str:
        full_tag = f"{self.prefix}{self.tag}" if self.prefix else self.tag
        return f"TagMatch({full_tag}, score={self.score:.2f}, source={self.source})"


@dataclass
class TagEmbedding:
    """Embedding entry for a HED tag."""

    tag: str
    long_form: str
    prefix: str
    vector: list[float] = field(repr=False)


@dataclass
class KeywordEmbedding:
    """Embedding entry for a curated keyword (anchor)."""

    keyword: str
    targets: list[str]  # HED tags this keyword points to
    vector: list[float] = field(repr=False)


# Deterministic keyword index mapping common terms to HED tags
# Ported from hed-lsp embeddings.ts KEYWORD_INDEX
KEYWORD_INDEX: dict[str, list[str]] = {
    # =====================
    # LAB ANIMALS (neuroscience research)
    # =====================
    # Primates
    "monkey": ["Animal", "Animal-agent"],
    "marmoset": ["Animal", "Animal-agent"],
    "macaque": ["Animal", "Animal-agent"],
    "rhesus": ["Animal", "Animal-agent"],
    "chimp": ["Animal", "Animal-agent"],
    "chimpanzee": ["Animal", "Animal-agent"],
    "primate": ["Animal", "Animal-agent"],
    "ape": ["Animal", "Animal-agent"],
    # Rodents
    "mouse": ["Animal", "Animal-agent", "Computer-mouse"],
    "mice": ["Animal", "Animal-agent"],
    "rat": ["Animal", "Animal-agent"],
    "rodent": ["Animal", "Animal-agent"],
    "hamster": ["Animal", "Animal-agent"],
    "gerbil": ["Animal", "Animal-agent"],
    "guinea": ["Animal", "Animal-agent"],
    # Other lab animals
    "ferret": ["Animal", "Animal-agent"],
    "rabbit": ["Animal", "Animal-agent"],
    "cat": ["Animal", "Animal-agent"],
    "dog": ["Animal", "Animal-agent"],
    "horse": ["Animal", "Animal-agent"],
    "pig": ["Animal", "Animal-agent"],
    "sheep": ["Animal", "Animal-agent"],
    "cow": ["Animal", "Animal-agent"],
    "goat": ["Animal", "Animal-agent"],
    # Model organisms
    "zebrafish": ["Animal", "Animal-agent"],
    "drosophila": ["Animal", "Animal-agent"],
    "fly": ["Animal", "Animal-agent"],
    "worm": ["Animal", "Animal-agent"],
    "elegans": ["Animal", "Animal-agent"],
    # General animal terms
    "animal": ["Animal", "Animal-agent"],
    "creature": ["Animal", "Animal-agent", "Organism"],
    "beast": ["Animal", "Animal-agent"],
    "mammal": ["Animal", "Animal-agent"],
    "bird": ["Animal", "Animal-agent"],
    "fish": ["Animal", "Animal-agent"],
    "pet": ["Animal", "Animal-agent"],
    # =====================
    # HUMAN PARTICIPANTS
    # =====================
    "subject": ["Human-agent", "Experiment-participant"],
    "participant": ["Human-agent", "Experiment-participant"],
    "volunteer": ["Human-agent", "Experiment-participant"],
    "patient": ["Human-agent", "Experiment-participant"],
    "person": ["Human", "Human-agent"],
    "people": ["Human", "Human-agent"],
    "human": ["Human", "Human-agent"],
    "man": ["Human", "Human-agent"],
    "woman": ["Human", "Human-agent"],
    "child": ["Human", "Human-agent"],
    "adult": ["Human", "Human-agent"],
    "infant": ["Human", "Human-agent"],
    "baby": ["Human", "Human-agent"],
    "toddler": ["Human", "Human-agent"],
    "adolescent": ["Human", "Human-agent"],
    "teenager": ["Human", "Human-agent"],
    "elderly": ["Human", "Human-agent"],
    # =====================
    # EXPERIMENTAL PARADIGM TERMS
    # =====================
    # Stimuli
    "stimulus": ["Experimental-stimulus", "Sensory-event"],
    "stimuli": ["Experimental-stimulus", "Sensory-event"],
    "stim": ["Experimental-stimulus", "Sensory-event"],
    "target": ["Target", "Experimental-stimulus"],
    "distractor": ["Distractor", "Experimental-stimulus"],
    "probe": ["Experimental-stimulus", "Cue"],
    "prime": ["Experimental-stimulus", "Cue"],
    "mask": ["Experimental-stimulus"],
    "flanker": ["Distractor", "Experimental-stimulus"],
    # Trial structure
    "trial": ["Experimental-trial"],
    "block": ["Time-block"],
    "run": ["Time-block"],
    "session": ["Time-block"],
    "epoch": ["Time-block"],
    # Timing
    "onset": ["Onset"],
    "offset": ["Offset"],
    "duration": ["Duration"],
    "delay": ["Delay"],
    "iti": ["Experimental-intertrial"],
    "isi": ["Experimental-intertrial"],
    "soa": ["Delay"],
    # Cues and instructions
    "cue": ["Cue", "Experimental-stimulus"],
    "go": ["Go-signal", "Cue"],
    "nogo": ["Cue", "Experimental-stimulus"],
    "stop": ["Cue", "Halt"],
    "instruction": ["Instructional"],
    "prompt": ["Cue", "Instructional"],
    # Responses
    "response": ["Participant-response"],
    "answer": ["Participant-response"],
    "reaction": ["Participant-response"],
    "rt": ["Participant-response"],
    # Feedback
    "feedback": ["Feedback"],
    "correct": ["Feedback"],
    "incorrect": ["Feedback"],
    "error": ["Feedback"],
    "accuracy": ["Feedback"],
    # =====================
    # REWARD & MOTIVATION
    # =====================
    "reward": ["Reward"],
    "punishment": ["Feedback"],
    "reinforcement": ["Reward", "Feedback"],
    "incentive": ["Reward"],
    "juice": ["Reward", "Drink"],
    "sugar": ["Reward", "Sweet"],
    "sucrose": ["Reward", "Sweet"],
    "money": ["Reward"],
    "monetary": ["Reward"],
    "win": ["Reward"],
    "loss": ["Feedback"],
    "gain": ["Reward"],
    # =====================
    # COGNITIVE STATES & PROCESSES
    # =====================
    # Attention
    "attention": ["Attentive", "Focused-attention"],
    "attentive": ["Attentive"],
    "focus": ["Focused-attention", "Attentive"],
    "focused": ["Focused-attention"],
    "concentrate": ["Focused-attention", "Attentive"],
    "distracted": ["Distracted"],
    "vigilance": ["Attentive", "Alert"],
    "orienting": ["Orienting-attention"],
    "covert": ["Covert-attention"],
    "overt": ["Overt-attention"],
    # Alertness/Arousal
    "alert": ["Alert"],
    "awake": ["Awake"],
    "asleep": ["Asleep"],
    "sleep": ["Asleep"],
    "drowsy": ["Drowsy"],
    "aroused": ["Aroused"],
    "arousal": ["Aroused"],
    # Rest/Baseline
    "rest": ["Rest", "Resting"],
    "resting": ["Resting", "Rest"],
    "baseline": ["Rest"],
    "fixation": ["Fixate"],
    "fixate": ["Fixate"],
    # Memory-related
    "remember": ["Attentive"],
    "recall": ["Attentive"],
    "encode": ["Attentive"],
    "retrieve": ["Attentive"],
    # =====================
    # EMOTIONAL STATES
    # =====================
    "happy": ["Happy"],
    "sad": ["Sad"],
    "angry": ["Angry"],
    "fear": ["Fearful"],
    "fearful": ["Fearful"],
    "afraid": ["Fearful"],
    "scared": ["Fearful"],
    "disgusted": ["Disgusted"],
    "disgust": ["Disgusted"],
    "surprised": ["Excited"],
    "neutral": ["Emotionally-neutral"],
    "emotional": ["Agent-emotional-state"],
    "emotion": ["Agent-emotional-state"],
    "mood": ["Agent-emotional-state"],
    "anxious": ["Stressed", "Fearful"],
    "stressed": ["Stressed"],
    "relaxed": ["Content", "Resting"],
    "excited": ["Excited"],
    "frustrated": ["Frustrated"],
    "bored": ["Passive"],
    # =====================
    # SENSORY MODALITIES
    # =====================
    # Visual
    "visual": ["See", "Visual-presentation"],
    "see": ["See"],
    "look": ["See", "Fixate"],
    "watch": ["See"],
    "view": ["See", "Visual-presentation"],
    "image": ["Image", "Visual-presentation"],
    "picture": ["Image", "Photograph"],
    "photo": ["Photograph", "Image"],
    "photograph": ["Photograph"],
    "video": ["Audiovisual-clip"],
    "movie": ["Audiovisual-clip"],
    "face": ["Face", "Move-face"],
    "scene": ["Image", "Visual-presentation"],
    "flash": ["Visual-presentation", "Sensory-event"],
    "flicker": ["Visual-presentation"],
    # Auditory
    "auditory": ["Hear", "Auditory-presentation"],
    "hear": ["Hear"],
    "listen": ["Hear"],
    "sound": ["Sound"],
    "audio": ["Sound", "Auditory-presentation"],
    "tone": ["Tone", "Sound"],
    "beep": ["Beep", "Sound"],
    "noise": ["Sound", "Signal-noise"],
    "music": ["Musical-sound"],
    "speech": ["Vocalized-sound", "Communicate-vocally"],
    "voice": ["Vocalized-sound"],
    "click": ["Sound", "Beep", "Press", "Push-button"],
    # Tactile/Somatosensory
    "touch": ["Touch", "Sense-by-touch"],
    "tactile": ["Tactile-presentation", "Sense-by-touch"],
    "vibration": ["Tactile-vibration"],
    "pressure": ["Tactile-pressure"],
    "pain": ["Pain"],
    "painful": ["Pain"],
    "thermal": ["Tactile-temperature"],
    "temperature": ["Tactile-temperature"],
    "hot": ["Tactile-temperature"],
    "cold": ["Tactile-temperature"],
    # Other senses
    "smell": ["Smell", "Olfactory-presentation"],
    "odor": ["Smell", "Olfactory-presentation"],
    "taste": ["Taste", "Gustatory-presentation"],
    "sweet": ["Sweet", "Taste"],
    "bitter": ["Bitter", "Taste"],
    "salty": ["Salty", "Taste"],
    "sour": ["Sour", "Taste"],
    # =====================
    # MOTOR ACTIONS & RESPONSES
    # =====================
    # Eye movements
    "saccade": ["Saccade", "Move-eyes"],
    "blink": ["Blink"],
    "gaze": ["Fixate", "Move-eyes"],
    "eye": ["Move-eyes", "Eye"],
    "pupil": ["Eye"],
    # Hand/Button responses
    "button": ["Push-button", "Press"],
    "press": ["Press", "Push-button"],
    "keypress": ["Press", "Push-button"],
    "tap": ["Press", "Touch"],
    "grip": ["Grasp"],
    "grasp": ["Grasp"],
    "reach": ["Move-body-part", "Move-upper-extremity"],
    "point": ["Move-upper-extremity"],
    # Body movements
    "walk": ["Walk"],
    "move": ["Move", "Move-body"],
    "movement": ["Move", "Move-body"],
    "motion": ["Move", "Move-body"],
    "gesture": ["Communicate-gesturally"],
    "nod": ["Nod-head"],
    "head": ["Head", "Move-head"],
    # Speech production
    "speak": ["Communicate-vocally", "Vocalize"],
    "say": ["Communicate-vocally"],
    "vocalize": ["Vocalize"],
    "articulate": ["Communicate-vocally"],
    # =====================
    # EQUIPMENT & DEVICES
    # =====================
    "screen": ["Computer-screen", "Display-device"],
    "monitor": ["Computer-screen", "Display-device"],
    "display": ["Display-device", "Computer-screen"],
    "headphones": ["Headphones"],
    "earphones": ["Headphones"],
    "speaker": ["Loudspeaker"],
    "keyboard": ["Keyboard"],
    "joystick": ["Joystick"],
    "trackball": ["Trackball"],
    "touchscreen": ["Touchscreen"],
    # =====================
    # BRAIN & NEUROANATOMY
    # =====================
    "brain": ["Brain"],
    "cortex": ["Brain", "Brain-region"],
    "frontal": ["Frontal-lobe", "Brain-region"],
    "parietal": ["Parietal-lobe", "Brain-region"],
    "temporal": ["Temporal-lobe", "Brain-region"],
    "occipital": ["Occipital-lobe", "Brain-region"],
    "cerebellum": ["Cerebellum", "Brain-region"],
    # =====================
    # BODY PARTS
    # =====================
    "hand": ["Hand"],
    "finger": ["Finger"],
    "arm": ["Arm"],
    "leg": ["Leg"],
    "foot": ["Foot"],
    "body": ["Body"],
    # =====================
    # CELLULAR & NETWORK NEUROSCIENCE
    # =====================
    "neuron": ["Brain", "Brain-region"],
    "cell": ["Brain", "Brain-region"],
    "spike": ["Data-feature", "Measurement-event"],
    "firing": ["Data-feature", "Measurement-event"],
    "unit": ["Data-feature"],
    "single-cell": ["Data-feature", "Measurement-event"],
    "multi-unit": ["Data-feature", "Measurement-event"],
    "network": ["Brain", "Brain-region"],
    "neural": ["Brain", "Brain-region"],
    "neuronal": ["Brain", "Brain-region"],
    "circuit": ["Brain", "Brain-region"],
    "ensemble": ["Brain", "Brain-region"],
    "lfp": ["Data-feature", "Measurement-event"],
    "oscillation": ["Data-feature"],
    "gamma": ["Data-feature"],
    "theta": ["Data-feature"],
    "alpha": ["Data-feature"],
    "beta": ["Data-feature"],
    "delta": ["Data-feature"],
    # =====================
    # RECORDING MODALITIES & NEUROIMAGING
    # =====================
    "eeg": ["Measurement-event", "Data-feature"],
    "meg": ["Measurement-event", "Data-feature"],
    "fmri": ["Measurement-event", "Data-feature"],
    "mri": ["Measurement-event"],
    "pet-scan": ["Measurement-event"],
    "nirs": ["Measurement-event", "Data-feature"],
    "fnirs": ["Measurement-event", "Data-feature"],
    "electrophysiology": ["Measurement-event", "Data-feature"],
    "imaging": ["Measurement-event"],
    "recording": ["Measurement-event", "Data-feature"],
    "scan": ["Measurement-event"],
    "acquisition": ["Measurement-event"],
    "trigger": ["Cue", "Experimental-stimulus"],
    "pulse": ["Sensory-event", "Measurement-event"],
    "tr": ["Time-block"],
    # =====================
    # NATURALISTIC & ECOLOGICAL PARADIGMS
    # =====================
    "naturalistic": ["Sensory-event", "Experimental-stimulus"],
    "ecological": ["Sensory-event"],
    "real-world": ["Sensory-event"],
    "free-viewing": ["See", "Sensory-event"],
    "narrative": ["Audiovisual-clip", "Sensory-event"],
    "story": ["Audiovisual-clip", "Hear"],
    "social": ["Human-agent", "Sensory-event"],
    "interaction": ["Communicate", "Agent-action"],
    "conversation": ["Communicate-vocally", "Hear"],
    "dialogue": ["Communicate-vocally", "Hear"],
    # =====================
    # GENERAL OBJECTS & PLACES
    # =====================
    "house": ["Building"],
    "home": ["Building"],
    "building": ["Building"],
    "room": ["Room"],
    "office": ["Building"],
    "lab": ["Building", "Room"],
    "laboratory": ["Building", "Room"],
    "car": ["Vehicle"],
    "vehicle": ["Vehicle"],
    "chair": ["Furniture"],
    "table": ["Furniture"],
    "desk": ["Furniture"],
    "food": ["Food"],
    "fruit": ["Fruit"],
    "apple": ["Apple", "Fruit"],
    "banana": ["Banana", "Fruit"],
    # =====================
    # SPATIAL RELATIONS
    # =====================
    "left-of": ["Left-of"],
    "right-of": ["Right-of"],
    "near": ["Near-to"],
    "far": ["Far-from"],
    "adjacent": ["Adjacent-to"],
    "inside": ["Inside"],
    "outside": ["Outside"],
    "in-front": ["In-front-of"],
    "beside": ["Beside"],
    "center": ["Center-of"],
    "top": ["Top-of"],
    "bottom": ["Bottom-of"],
    "under": ["Under"],
    # =====================
    # TEMPORAL RELATIONS
    # =====================
    "simultaneous": ["Synchronous-with"],
    "synchronous": ["Synchronous-with"],
    "asynchronous": ["Asynchronous-with"],
    "concurrent": ["Synchronous-with"],
    "sequential": ["After"],
    "following": ["After"],
    "preceding": ["Before"],
    # =====================
    # EXTENDED BODY PARTS
    # =====================
    "lips": ["Lip"],
    "jaw": ["Jaw"],
    "scalp": ["Hair"],
    "chest": ["Torso"],
    "back": ["Torso"],
    "stomach": ["Abdomen"],
    # =====================
    # DATA ARTIFACTS & SIGNAL QUALITY
    # =====================
    "artifact": ["Artifact"],
    "artefact": ["Artifact"],
    "noise-artifact": ["Artifact"],
    "muscle-artifact": ["Artifact"],
    "electrode-artifact": ["Artifact"],
    "noisy": ["Artifact"],
    "clean": ["Normal"],
    "corrupted": ["Artifact"],
    # =====================
    # SCORE LIBRARY - EEG/CLINICAL (sc: prefix)
    # =====================
    "rem": ["sc:Sleep-stage-R"],
    "rem-sleep": ["sc:Sleep-stage-R"],
    "nrem": ["sc:Sleep-stage-N1", "sc:Sleep-stage-N2", "sc:Sleep-stage-N3"],
    "stage-1": ["sc:Sleep-stage-N1"],
    "stage-2": ["sc:Sleep-stage-N2"],
    "stage-3": ["sc:Sleep-stage-N3"],
    "n1": ["sc:Sleep-stage-N1"],
    "n2": ["sc:Sleep-stage-N2"],
    "n3": ["sc:Sleep-stage-N3"],
    "slow-wave-sleep": ["sc:Sleep-stage-N3"],
    "deep-sleep": ["sc:Sleep-stage-N3"],
    "spindle": ["sc:Sleep-spindle"],
    "sleep-spindle": ["sc:Sleep-spindle"],
    "k-complex": ["sc:K-complex"],
    "alpha-rhythm": ["sc:Alpha-activity"],
    "beta-rhythm": ["sc:Beta-activity"],
    "theta-rhythm": ["sc:Theta-activity"],
    "delta-rhythm": ["sc:Delta-activity"],
    "alpha-activity": ["sc:Alpha-activity"],
    "beta-activity": ["sc:Beta-activity"],
    "theta-activity": ["sc:Theta-activity"],
    "delta-activity": ["sc:Delta-activity"],
    "seizure": ["sc:Seizure"],
    "epileptic": ["sc:Epileptiform-activity"],
    "epileptiform": ["sc:Epileptiform-activity"],
    "spike-wave": ["sc:Spike-and-wave"],
    "sharp-wave": ["sc:Sharp-wave"],
    "interictal": ["sc:Interictal-finding"],
    "ictal": ["sc:Ictal-finding"],
    # =====================
    # ENVIRONMENTAL CONTEXT
    # =====================
    "outdoor": ["Outdoors"],
    "indoor": ["Indoors"],
    "virtual-reality": ["Virtual-world"],
    "vr": ["Virtual-world"],
    "ar": ["Augmented-reality"],
    "underwater": ["Underwater"],
    "daytime": ["Daytime"],
    "nighttime": ["Nighttime"],
    "day": ["Daytime"],
    "night": ["Nighttime"],
    # =====================
    # INFORMATIONAL PROPERTIES
    # =====================
    "difficult": ["Difficult"],
    "easy": ["Easy"],
    "hard": ["Difficult"],
    "predictable": ["Expected"],
    "unpredictable": ["Unexpected"],
    "meaningless": ["Nonsensical"],
    "nonsense": ["Nonsensical"],
    # =====================
    # COGNITIVE ACTIONS
    # =====================
    "decide": ["Decide"],
    "decision": ["Decide"],
    "choose": ["Decide"],
    "choice": ["Decide"],
    "imagining": ["Imagine"],
    "mental-imagery": ["Imagine"],
    "prediction": ["Predict"],
    "expect": ["Expect"],
    "expecting": ["Expect"],
    "anticipate": ["Expect"],
    "counting": ["Count"],
    "calculate": ["Count"],
    "estimate": ["Estimate"],
    "judgment": ["Judge"],
    "attend": ["Attend-to"],
    "attending": ["Attend-to"],
    "notice": ["Attend-to"],
    "detecting": ["Detect"],
    "recognition": ["Recognize"],
    "categorize": ["Discriminate"],
    "compare": ["Compare"],
    "comparing": ["Compare"],
    "evaluate": ["Evaluate"],
    # =====================
    # LANGUAGE & LINGUISTIC TERMS
    # =====================
    "words": ["Word"],
    "sentences": ["Sentence"],
    "text": ["Character"],
    "letters": ["Character"],
    "morpheme": ["Morpheme"],
    "reading": ["Read"],
    "writing": ["Write"],
    "spelling": ["Spell"],
    "naming": ["Communicate-vocally"],
    "comprehension": ["Hear", "Read"],
    "language": ["Communicate"],
    "verbal": ["Communicate-vocally"],
    "nonverbal": ["Communicate-gesturally"],
    # =====================
    # COLORS
    # =====================
    "grey": ["Gray"],
    "colour": ["CSS-color"],
    "colored": ["CSS-color"],
    "coloured": ["CSS-color"],
    # =====================
    # VISUAL PROPERTIES
    # =====================
    "bright": ["High-contrast"],
    "dim": ["Low-contrast"],
    "contrast": ["High-contrast"],
    "luminous": ["High-contrast"],
    "dark": ["Low-contrast"],
    "opaque": ["Opaque"],
    "transparent": ["Transparent"],
    "blurry": ["Blurry"],
    "clear": ["Clear"],
    "monochrome": ["Grayscale"],
    # =====================
    # SHAPES & GEOMETRY
    # =====================
    "circle": ["Ellipse"],
    "circular": ["Ellipse"],
    "line": ["Line"],
    "checkerboard": ["Checkerboard"],
    "grating": ["Grating"],
    "gabor": ["Gabor-patch"],
    "dot": ["Ellipse"],
    # =====================
    # ADDITIONAL TASK TYPES
    # =====================
    "stroop": ["Experimental-stimulus"],
    "n-back": ["Experimental-stimulus"],
    "nback": ["Experimental-stimulus"],
    "working-memory": ["Attentive"],
    "memory-task": ["Experimental-stimulus"],
    "detection": ["Experimental-stimulus", "Target"],
    "discrimination": ["Experimental-stimulus"],
    "localization": ["Experimental-stimulus"],
    "search": ["Experimental-stimulus", "See"],
    "visual-search": ["Experimental-stimulus", "See"],
    "antisaccade": ["Saccade", "Experimental-stimulus"],
    "prosaccade": ["Saccade", "Experimental-stimulus"],
    "pursuit": ["Move-eyes"],
    "smooth-pursuit": ["Move-eyes"],
    # =====================
    # COMMUNICATION & SOCIAL
    # =====================
    "communication": ["Communicate"],
    "talk": ["Communicate-vocally"],
    "talking": ["Communicate-vocally"],
    "chat": ["Communicate-vocally"],
    "sign": ["Communicate-gesturally"],
    "signing": ["Communicate-gesturally"],
    "facial-expression": ["Move-face"],
    "expression": ["Move-face"],
}


class SemanticSearchManager:
    """Manages semantic search for HED tags using embeddings and keyword matching.

    Implements a dual-embedding architecture:
    1. Keyword index - deterministic lookup for known terms
    2. Embedding search - semantic similarity for natural language queries

    When both sources agree on a tag, confidence is boosted.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        embeddings_path: Path | None = None,
    ) -> None:
        """Initialize the semantic search manager.

        Args:
            model_id: HuggingFace model ID for embeddings
            embeddings_path: Path to pre-computed embeddings JSON file
        """
        self.model_id = model_id
        self.embeddings_path = embeddings_path
        self._model: SentenceTransformer | None = None
        self._tag_embeddings: dict[str, TagEmbedding] = {}
        self._keyword_embeddings: list[KeywordEmbedding] = []
        self._embeddings_loaded = False
        self._dimensions = 1024  # Qwen3-Embedding default

    def _get_model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.model_id}")
            self._model = SentenceTransformer(self.model_id)
            logger.info("Embedding model loaded successfully")
        return self._model

    def load_embeddings(self, path: Path | None = None) -> bool:
        """Load pre-computed embeddings from file.

        Args:
            path: Path to embeddings JSON file (overrides constructor path)

        Returns:
            True if embeddings loaded successfully
        """
        embeddings_path = path or self.embeddings_path
        if embeddings_path is None or not embeddings_path.exists():
            logger.warning(f"Embeddings file not found: {embeddings_path}")
            return False

        try:
            with open(embeddings_path) as f:
                data = json.load(f)

            # Load tag embeddings
            for entry in data.get("tags", []):
                key = f"{entry.get('prefix', '')}{entry['tag']}".lower()
                self._tag_embeddings[key] = TagEmbedding(
                    tag=entry["tag"],
                    long_form=entry.get("long_form", entry["tag"]),
                    prefix=entry.get("prefix", ""),
                    vector=entry["vector"],
                )

            # Load keyword embeddings
            for entry in data.get("keywords", []):
                self._keyword_embeddings.append(
                    KeywordEmbedding(
                        keyword=entry["keyword"],
                        targets=entry["targets"],
                        vector=entry["vector"],
                    )
                )

            self._dimensions = data.get("dimensions", 1024)
            self._embeddings_loaded = True
            logger.info(
                f"Loaded {len(self._tag_embeddings)} tag embeddings and "
                f"{len(self._keyword_embeddings)} keyword embeddings"
            )
            return True

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to load embeddings: {e}")
            return False

    def find_by_keyword(self, query: str) -> list[TagMatch]:
        """Find tags matching a keyword from the deterministic index.

        Args:
            query: Keyword to look up

        Returns:
            List of matching tags with high confidence (0.95)
        """
        normalized = query.lower().strip()
        matching_tags = KEYWORD_INDEX.get(normalized)

        if not matching_tags:
            return []

        results = []
        for tag_name in matching_tags:
            # Parse prefix if present (e.g., "sc:Seizure")
            prefix = ""
            tag = tag_name
            if ":" in tag_name:
                prefix, tag = tag_name.split(":", 1)
                prefix = f"{prefix}:"

            results.append(
                TagMatch(
                    tag=tag,
                    long_form=tag,  # Could be enhanced with schema lookup
                    prefix=prefix,
                    score=0.95,  # High confidence for exact keyword match
                    source="keyword",
                )
            )

        return results

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors.

        Args:
            a: First vector (should be normalized)
            b: Second vector (should be normalized)

        Returns:
            Cosine similarity score
        """
        # Vectors should already be normalized from sentence-transformers
        return float(np.dot(a, b))

    def embed(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as numpy array
        """
        model = self._get_model()
        return model.encode(text.lower(), normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Array of embedding vectors
        """
        model = self._get_model()
        return model.encode(
            [t.lower() for t in texts],
            normalize_embeddings=True,
        )

    def find_similar(
        self,
        keywords: list[str],
        top_k: int = 10,
        use_embeddings: bool = True,
    ) -> list[TagMatch]:
        """Find semantically similar tags using dual-embedding architecture.

        Algorithm:
        1. Check keyword index first for exact matches
        2. Search keyword embeddings to collect votes for target tags
        3. Search tag embeddings directly
        4. Combine evidence - tags from both sources get boosted

        Args:
            keywords: List of keywords to search for
            top_k: Maximum number of results to return
            use_embeddings: Whether to use embedding search (requires model)

        Returns:
            List of matched tags sorted by score
        """
        # Collect all matches from keyword index first
        keyword_matches: dict[str, TagMatch] = {}

        for keyword in keywords:
            for match in self.find_by_keyword(keyword):
                key = f"{match.prefix}{match.tag}".lower()
                if key not in keyword_matches or match.score > keyword_matches[key].score:
                    keyword_matches[key] = match

        # If no embeddings loaded or disabled, return keyword matches only
        if not use_embeddings or not self._embeddings_loaded:
            results = list(keyword_matches.values())
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]

        # Embedding-based search
        tag_votes: dict[str, dict] = {}  # tag -> {votes, max_similarity}
        direct_matches: dict[str, dict] = {}  # tag -> {entry, similarity}

        KEYWORD_THRESHOLD = 0.6
        TAG_THRESHOLD = 0.5
        TOP_KEYWORDS = 10

        for keyword in keywords:
            query_embedding = self.embed(keyword)

            # Search keyword embeddings - collect votes
            if self._keyword_embeddings:
                kw_similarities = []
                for kw in self._keyword_embeddings:
                    kw_vector = np.array(kw.vector)
                    sim = self._cosine_similarity(query_embedding, kw_vector)
                    if sim >= KEYWORD_THRESHOLD:
                        kw_similarities.append((kw, sim))

                kw_similarities.sort(key=lambda x: x[1], reverse=True)

                for kw, sim in kw_similarities[:TOP_KEYWORDS]:
                    for target in kw.targets:
                        if target in tag_votes:
                            tag_votes[target]["votes"] += 1
                            tag_votes[target]["max_sim"] = max(tag_votes[target]["max_sim"], sim)
                        else:
                            tag_votes[target] = {"votes": 1, "max_sim": sim}

            # Search tag embeddings directly
            for key, entry in self._tag_embeddings.items():
                tag_vector = np.array(entry.vector)
                sim = self._cosine_similarity(query_embedding, tag_vector)
                if sim >= TAG_THRESHOLD:
                    if key not in direct_matches or sim > direct_matches[key]["similarity"]:
                        direct_matches[key] = {"entry": entry, "similarity": sim}

        # Combine evidence
        combined: dict[str, TagMatch] = {}

        # Add tags from keyword votes
        for tag_name, vote_data in tag_votes.items():
            # Parse prefix
            prefix = ""
            tag = tag_name
            if ":" in tag_name:
                prefix, tag = tag_name.split(":", 1)
                prefix = f"{prefix}:"

            key = f"{prefix}{tag}".lower()
            direct_sim = direct_matches.get(key, {}).get("similarity", 0)

            # Combined score: keyword similarity with vote boost + direct similarity
            import math

            kw_score = vote_data["max_sim"] * (1 + math.log(vote_data["votes"] + 1) * 0.2)
            boost = 1.5 if direct_sim > 0 else 1.0
            combined_score = kw_score * boost + direct_sim * 0.3

            source: Literal["keyword", "embedding", "both"] = (
                "both" if direct_sim > 0 else "embedding"
            )

            combined[key] = TagMatch(
                tag=tag,
                long_form=tag,
                prefix=prefix,
                score=min(combined_score, 0.94),  # Cap below exact keyword match
                source=source,
            )

        # Add direct matches not in keyword votes
        for key, data in direct_matches.items():
            if key not in combined:
                entry = data["entry"]
                combined[key] = TagMatch(
                    tag=entry.tag,
                    long_form=entry.long_form,
                    prefix=entry.prefix,
                    score=data["similarity"],
                    source="embedding",
                )

        # Merge with keyword index matches (keyword matches take precedence)
        for key, match in keyword_matches.items():
            if key in combined:
                # Boost score if also found via embeddings
                combined[key] = TagMatch(
                    tag=match.tag,
                    long_form=match.long_form,
                    prefix=match.prefix,
                    score=min(match.score + 0.02, 0.97),  # Slight boost
                    source="both",
                )
            else:
                combined[key] = match

        results = list(combined.values())
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def is_available(self) -> bool:
        """Check if semantic search is available (embeddings loaded)."""
        return self._embeddings_loaded

    def get_stats(self) -> dict:
        """Get statistics about loaded embeddings."""
        return {
            "tag_embeddings": len(self._tag_embeddings),
            "keyword_embeddings": len(self._keyword_embeddings),
            "keyword_index_size": len(KEYWORD_INDEX),
            "dimensions": self._dimensions,
            "model_id": self.model_id,
            "embeddings_loaded": self._embeddings_loaded,
        }


# Global instance for shared use
_default_manager: SemanticSearchManager | None = None


def get_semantic_search_manager() -> SemanticSearchManager:
    """Get the default global semantic search manager instance."""
    global _default_manager
    if _default_manager is None:
        _default_manager = SemanticSearchManager()
    return _default_manager
