import json
import numpy as np
from pathlib import Path
from classification.model_training import evaluate_single_model
from data.data_configs import SUBJECTS


def load_optimal_parameters(split_strategy: str):
    base = Path(__file__).parent / 'results'
    fname = f"optimal_parameters_{split_strategy}_latest.json"
    file = base / fname
    # fallback for different naming conventions
    if not file.exists() and 'leave' in split_strategy:
        alt = base / 'optimal_parameters_leave_n_out_latest.json'
        if alt.exists():
            file = alt
    if not file.exists():
        file = base / 'optimal_parameters_normal_latest.json'
    with open(file, 'r') as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            # File exists but is empty or invalid JSON — return empty dict
            return {}
    
def make_jsonable(o):
    # Recursively convert objects to JSON-serializable Python types
    if isinstance(o, dict):
        return {k: make_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [make_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return make_jsonable(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def select_best_models(opt_results: dict):
    out = {}
    # features
    if 'features_rf' in opt_results and 'features_svm' in opt_results:
        out['features_rf' if opt_results['features_rf']['score'] > opt_results['features_svm']['score'] else 'features_svm'] = \
            opt_results['features_rf' if opt_results['features_rf']['score'] > opt_results['features_svm']['score'] else 'features_svm']
    # et
    if 'et_rf' in opt_results and 'et_svm' in opt_results:
        out['et_rf' if opt_results['et_rf']['score'] > opt_results['et_svm']['score'] else 'et_svm'] = \
            opt_results['et_rf' if opt_results['et_rf']['score'] > opt_results['et_svm']['score'] else 'et_svm']
    # add remaining
    for name, v in opt_results.items():
        if not (name.startswith('features_') or name.startswith('et_')):
            out[name] = v
    return out


def evaluate_and_save(models: dict, split_strategy: str, subjects, out_filename: str):
    out_file = Path(__file__).parent / 'results' / out_filename
    results = {}
    if out_file.exists():
        with open(out_file, 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    for model_name, model_data in models.items():
        params = model_data['params']
        # Flatten any single-item lists to scalars
        flat_params = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v) for k, v in params.items()}
        res = evaluate_single_model(model_name, subjects, flat_params, split_strategy=split_strategy)
        
        current_score = res.get("balanced_acc", res.get("score", 0))
        
        # Check if we should update the result
        should_save = True
        if model_name in results:
            old_score = results[model_name].get('balanced_acc', results[model_name].get('score', 0))
            if current_score <= old_score:
                should_save = False
                print(f"  ⚠️  {model_name}: New score {current_score:.4f} not better than existing {old_score:.4f}, skipping")

        if should_save:
            # Save only essential metrics, not full results
            results[model_name] = {
                "score": current_score,
                "balanced_acc": current_score,
                "params": flat_params
            }
            print(f"  💾 Updated {model_name} with score {current_score:.4f}")
            
            # Save immediately after each update
            with open(out_file, 'w') as f:
                json.dump(make_jsonable(results), f, indent=2)

# Run for both strategies
for strategy, out_file in [('normal', 'evaluation_results_normal.json'), ('leave_one_out', 'evaluation_results_leave_one_out.json')]:
    optimal = load_optimal_parameters(strategy)
    to_eval = select_best_models(optimal)

    params =  {"et_rf": {
    "params": {
      "max_depth": 30,
      "min_samples_leaf": 2,
      "min_samples_split": 5,
      "n_estimators": 200
    }
    }}

    # params =  {'cbramod': {
    # 'params':{
    #     'dropout': 0.1,
    #     'hidden_dim': 200,
    #     'weight_decay': 0.05,
    #     'lr': 0.0001,
    #     'label_smoothing': 0.0
    # }}}

    evaluate_and_save(to_eval, strategy, SUBJECTS, out_file)
