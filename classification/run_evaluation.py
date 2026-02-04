import json
from pathlib import Path
from classification.training_utils import evaluate_single_model
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
        return json.load(f)


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
            results = json.load(f)
    for model_name, model_data in models.items():
        params = model_data['params']
        # Flatten any single-item lists to scalars
        flat_params = {k: (v[0] if isinstance(v, (list, tuple)) and len(v) == 1 else v) for k, v in params.items()}
        res = evaluate_single_model(model_name, subjects, flat_params, split_strategy=split_strategy)
        results[model_name] = res
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)


# Run for both strategies
for strategy, out_file in [('normal', 'evaluation_results_normal.json'), ('leave_one_out', 'evaluation_results_leave_one_out.json')]:
    optimal = load_optimal_parameters(strategy)
    to_eval = select_best_models(optimal)

    params = {
    'cbramod': {
        "params": {
        # Regularization & capacity
        'dropout': 0.1,
        'hidden_dim': 200,
        'weight_decay': 0.05,
        'lr': 0.0001,
    }
    }
}
    evaluate_and_save(params, strategy, SUBJECTS[:1], out_file)


