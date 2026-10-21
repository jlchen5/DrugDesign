"""A reusable script template: train, persist and screen candidates.

Run after installation: python examples/predict_script.py
Replace the example paths and estimator with your assay-specific data/model.
"""

from pathlib import Path

from drugdesign import PipelineConfig, load_config, run_pipeline


def main():
    cfg: PipelineConfig = load_config(Path(__file__).with_name("config.toml"))
    cfg.model.params = {"n_estimators": 100, "n_jobs": -1}
    result = run_pipeline(cfg)
    print(result.report.summary())
    print(result.predictions.to_string(index=False))


if __name__ == "__main__":
    main()
