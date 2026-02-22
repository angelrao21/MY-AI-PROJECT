"""Reasoner combining rule-based logic, simple fuzzy membership, and CSP-like checks.

Provides `decide_green_time` which takes state + predictor output and returns
an allocated green time and an insights dict for human-AI interaction.
"""
from typing import Dict, Any, List
import json
import os

# Try to load knowledge (parameters) from nearby file
def _load_knowledge():
    path = os.path.join(os.path.dirname(__file__), 'knowledge.json')
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        # default parameters
        return {
            'min_green': 10,
            'max_green': 60,
            'cycle_limit': 200,
            'density_thresholds': [5, 15],
            'allocation_factor': 1.0
        }

KNOW = _load_knowledge()

def _fuzzy_density_label(arrivals: float, thresholds: List[float]):
    low, high = thresholds
    if arrivals <= low:
        return 'low'
    if arrivals >= high:
        return 'high'
    return 'medium'

def decide_green_time(state: Dict[str, Any], predictor, direction_index: int) -> Dict[str, Any]:
    """Decide a green time for `direction_index` using state and predictor.

    Returns a dict: { 'green_time': int, 'predicted': float, 'label': str, 'explanation': str }
    """
    min_g = KNOW.get('min_green', 10)
    max_g = KNOW.get('max_green', 60)
    factor = KNOW.get('allocation_factor', 1.0)
    thresholds = KNOW.get('density_thresholds', [5, 15])

    # get predicted arrivals for directions
    preds = predictor.predict() if hasattr(predictor, 'predict') else predictor
    if isinstance(preds, list) or isinstance(preds, tuple):
        pred = float(preds[direction_index])
    else:
        pred = float(preds)

    label = _fuzzy_density_label(pred, thresholds)

    # rule-based base allocation
    if label == 'low':
        green = min_g
    elif label == 'medium':
        green = min(int(min_g + factor * pred), max_g)
    else:
        green = min(int(min_g + factor * pred * 1.2), max_g)

    # simple CSP-like check: do not exceed cycle limit if provided
    cycle_limit = KNOW.get('cycle_limit', None)
    if cycle_limit is not None:
        # we don't have full cycle composition here; this is a conservative check
        green = min(green, cycle_limit)

    explanation = f"pred={pred:.2f}, label={label}, base_alloc={green}"

    return {
        'green_time': int(max(min_g, min(green, max_g))),
        'predicted': pred,
        'label': label,
        'explanation': explanation
    }
