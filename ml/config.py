from pathlib import Path

ML_DIR = Path(__file__).resolve().parent
ROOT_DIR = ML_DIR.parent

DATASET_DIR = ML_DIR / "dataset"
ARTIFACTS_DIR = ML_DIR / "artifacts"
LANDMARKS_FILE = ARTIFACTS_DIR / "landmarks.npz"
EXTRACTION_STATS_FILE = ARTIFACTS_DIR / "extraction_stats.json"
MODEL_FILE = ARTIFACTS_DIR / "gesture_classifier.keras"
METRICS_FILE = ARTIFACTS_DIR / "metrics.json"
FIGURES_DIR = ROOT_DIR / "docs" / "images"

ANDROID_ASSETS_DIR = ROOT_DIR / "android" / "app" / "src" / "main" / "assets"
HAND_LANDMARKER_MODEL = ANDROID_ASSETS_DIR / "hand_landmarker.task"
WEB_MODEL_DIR = ROOT_DIR / "web" / "model"
TEST_FIXTURES_DIR = ML_DIR / "tests" / "fixtures"

# Every folder in the dataset, with the Portuguese sign name, its English meaning and
# the hand shape the sign uses.
GESTURES = {
    "agua": {"name": "Água", "meaning": "Water", "hand_shape": "Flat hand, edge towards the camera"},
    "bom_dia": {"name": "Bom dia", "meaning": "Good morning", "hand_shape": "Thumbs up"},
    "nao": {"name": "Não", "meaning": "No", "hand_shape": "Index finger up"},
    "ola": {"name": "Olá", "meaning": "Hello", "hand_shape": "Open palm, fingers spread"},
    "por_favor": {"name": "Por favor", "meaning": "Please", "hand_shape": "Flat hand, fingers together"},
    "sim": {"name": "Sim", "meaning": "Yes", "hand_shape": "Closed fist"},
}

# Gestures the shipped model recognises (see the README for why some are left out).
ENABLED_GESTURES = ["agua", "bom_dia", "ola", "sim"]

# The apps average the classifier output over this many consecutive frames with a hand
# and only show a gesture when the averaged confidence reaches MIN_CONFIDENCE.
SMOOTHING_WINDOW = 10
MIN_CONFIDENCE = 0.6

SEED = 42
