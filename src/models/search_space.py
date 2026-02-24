import optuna


def get_saits_search_space(trial: optuna.Trial) -> dict:
    """Optuna search space for SAITS."""
    return {
        "n_layers": trial.suggest_categorical("n_layers", [2, 4]),
        "d_model": trial.suggest_categorical("d_model", [32, 64]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "d_ffn": trial.suggest_categorical("d_ffn", [64, 128]),
    }


def get_grud_search_space(trial: optuna.Trial) -> dict:
    """Optuna search space for GRUD."""
    return {
        "rnn_hidden_size": trial.suggest_categorical("rnn_hidden_size", [32, 64]),
    }


def get_usgan_search_space(trial: optuna.Trial) -> dict:
    """Optuna search space for USGAN."""
    return {
        "rnn_hidden_size": trial.suggest_categorical("rnn_hidden_size", [32, 64]),
    }


def get_itransformer_search_space(trial: optuna.Trial) -> dict:
    """Optuna search space for iTransformer."""
    return {
        "n_layers": trial.suggest_categorical("n_layers", [2, 4]),
        "d_model": trial.suggest_categorical("d_model", [32, 64]),
        "n_heads": trial.suggest_categorical("n_heads", [2, 4]),
        "d_ffn": trial.suggest_categorical("d_ffn", [64, 128]),
    }


def get_knn_search_space(trial: optuna.Trial) -> dict:
    """Optuna search space for KNN."""
    return {
        "knn_neighbors": trial.suggest_int("knn_neighbors", 3, 7, step=2),
    }


def has_search_space(model_name: str) -> bool:
    """Check if model supports HPO."""
    return model_name.lower() in {"saits", "grud", "usgan", "itransformer", "knn"}


def get_search_space_for_model(model_name: str, trial: optuna.Trial) -> dict | None:
    """Return specific hyperparameters sampled from trial, or None if HPO not supported."""
    model_name = model_name.lower()
    if model_name == "saits":
        return get_saits_search_space(trial)
    if model_name == "grud":
        return get_grud_search_space(trial)
    if model_name == "usgan":
        return get_usgan_search_space(trial)
    if model_name == "itransformer":
        return get_itransformer_search_space(trial)
    if model_name == "knn":
        return get_knn_search_space(trial)
    return None
