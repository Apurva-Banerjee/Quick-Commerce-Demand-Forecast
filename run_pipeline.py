"""CLI entrypoint for the demand forecasting project."""

from src.pipeline import run_pipeline


if __name__ == "__main__":
    results = run_pipeline()
    print("Pipeline completed.")
    print("Model comparison metrics:")
    for model_name, metric_values in results.items():
        print(f"\n{model_name.upper()}")
        for k, v in metric_values.items():
            if isinstance(v, (int, float)):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
