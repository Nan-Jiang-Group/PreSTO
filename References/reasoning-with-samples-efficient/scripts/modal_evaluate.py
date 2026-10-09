#!/usr/bin/env python3
"""
Modal app for evaluating SWE-bench predictions in the cloud.

This runs swebench evaluation in Modal's containerized environment,
handling the heavy lifting of cloning repos, applying patches, and running tests.

Usage:
    # Evaluate a single prediction file
    modal run scripts/modal_evaluate.py --prediction predictions/swebench_lite_model_strategy.jsonl
    
    # Evaluate all predictions in directory
    modal run scripts/modal_evaluate.py --predictions-dir predictions/
    
    # Evaluate with custom dataset
    modal run scripts/modal_evaluate.py --prediction predictions/file.jsonl --dataset princeton-nlp/SWE-bench_Verified
"""

import modal
import os
from pathlib import Path

# Create Modal app
app = modal.App("swebench-evaluator")

# Create Modal image with swebench installed
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install(
        "swebench",
        "datasets",
        "requests",
        "tqdm",
        "modal",  # Modal SDK for --modal true flag
    )
)

# Create volume for persistent storage
volume = modal.Volume.from_name("swebench-data", create_if_missing=True)

VOLUME_DIR = "/data"
PREDICTIONS_DIR = f"{VOLUME_DIR}/predictions"
RESULTS_DIR = f"{VOLUME_DIR}/results"
LOGS_DIR = f"{VOLUME_DIR}/logs"


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
    timeout=3600 * 4,  # 4 hours timeout
    cpu=16,  # 16 CPU cores (max speed)
    memory=131072,  # 128GB RAM (maximum parallelism)
)
def evaluate_prediction(
    prediction_file: str,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    max_workers: int = 4,
    modal_token_id: str = None,
    modal_token_secret: str = None,
):
    """
    Evaluate a single SWE-bench prediction file.
    
    Args:
        prediction_file: Path to prediction JSONL file in volume
        dataset: SWE-bench dataset to use
        max_workers: Number of parallel workers
    
    Returns:
        dict with evaluation results
    """
    import subprocess
    import json
    from pathlib import Path
    
    print(f"🚀 Starting evaluation for: {prediction_file}")
    print(f"📊 Dataset: {dataset}")
    print(f"⚙️  Workers: {max_workers}")
    print(f"☁️  Using Modal's cloud infrastructure for evaluation")
    
    # Set up Modal credentials if provided
    if modal_token_id and modal_token_secret:
        print("🔑 Configuring Modal credentials...")
        import os
        from pathlib import Path
        
        # Create .modal.toml file with credentials (directly in home directory)
        modal_config = f"""[shivaperi47]
token_id = "{modal_token_id}"
token_secret = "{modal_token_secret}"
active = true
"""
        
        modal_config_path = Path.home() / ".modal.toml"
        with open(modal_config_path, "w") as f:
            f.write(modal_config)
        
        print(f"✅ Modal credentials configured at {modal_config_path}")
    elif "MODAL_TOKEN_ID" in os.environ and "MODAL_TOKEN_SECRET" in os.environ:
        print("🔑 Configuring Modal credentials from environment...")
        
        modal_config = f"""[shivaperi47]
token_id = "{os.environ['MODAL_TOKEN_ID']}"
token_secret = "{os.environ['MODAL_TOKEN_SECRET']}"
active = true
"""
        
        modal_config_path = Path.home() / ".modal.toml"
        with open(modal_config_path, "w") as f:
            f.write(modal_config)
        
        print(f"✅ Modal credentials configured from environment at {modal_config_path}")
    else:
        print("⚠️  Warning: No Modal credentials provided. Evaluation may fail.")
    
    # Create output directories
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Determine output paths
    pred_name = Path(prediction_file).stem
    report_dir = f"{LOGS_DIR}/{pred_name}"
    result_file = f"{RESULTS_DIR}/{pred_name}_results.json"
    
    # Generate a unique run ID
    import time
    run_id = f"{pred_name}_{int(time.time())}"
    
    # Run swebench evaluation with Modal integration
    # The --modal true flag tells swebench to use Modal's infrastructure
    cmd = [
        "python", "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--predictions_path", prediction_file,
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--modal", "true",
    ]
    
    print(f"📝 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        
        print("✅ Evaluation completed successfully!")
        print(result.stdout)
        
        # Parse results from logs
        results_summary = {
            "prediction_file": prediction_file,
            "dataset": dataset,
            "status": "success",
            "stdout": result.stdout,
            "report_dir": report_dir,
        }
        
        # Try to extract pass rate from output
        for line in result.stdout.split('\n'):
            if "resolved" in line.lower() or "pass" in line.lower():
                print(f"📊 {line}")
        
        # Save results
        with open(result_file, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        # Commit to volume
        volume.commit()
        
        return results_summary
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed!")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        error_summary = {
            "prediction_file": prediction_file,
            "dataset": dataset,
            "status": "failed",
            "error": str(e),
            "stdout": e.stdout,
            "stderr": e.stderr,
        }
        
        # Save error info
        with open(result_file, 'w') as f:
            json.dump(error_summary, f, indent=2)
        
        volume.commit()
        
        raise


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
)
def upload_prediction(filename: str, content: str) -> str:
    """
    Upload a prediction file to Modal volume.
    
    Args:
        filename: Name of the prediction file
        content: Content of the prediction file
    
    Returns:
        Path in Modal volume
    """
    from pathlib import Path
    
    # Create predictions directory
    Path(PREDICTIONS_DIR).mkdir(parents=True, exist_ok=True)
    
    # Write to volume
    volume_path = f"{PREDICTIONS_DIR}/{filename}"
    
    with open(volume_path, 'w') as f:
        f.write(content)
    
    volume.commit()
    
    print(f"📤 Uploaded: {filename} -> {volume_path}")
    return volume_path


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
)
def download_results(prediction_name: str) -> dict:
    """
    Download evaluation results for a prediction.
    
    Args:
        prediction_name: Name of prediction file (without extension)
    
    Returns:
        dict with results
    """
    import json
    from pathlib import Path
    
    result_file = f"{RESULTS_DIR}/{prediction_name}_results.json"
    
    if not Path(result_file).exists():
        return {"error": f"Results not found: {result_file}"}
    
    with open(result_file, 'r') as f:
        results = json.load(f)
    
    print(f"📥 Downloaded results for: {prediction_name}")
    return results


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
)
def list_predictions() -> list:
    """List all prediction files in volume."""
    from pathlib import Path
    
    pred_dir = Path(PREDICTIONS_DIR)
    if not pred_dir.exists():
        return []
    
    predictions = [f.name for f in pred_dir.glob("*.jsonl")]
    return predictions


@app.function(
    image=image,
    volumes={VOLUME_DIR: volume},
)
def list_results() -> list:
    """List all result files in volume."""
    from pathlib import Path
    
    res_dir = Path(RESULTS_DIR)
    if not res_dir.exists():
        return []
    
    results = [f.name for f in res_dir.glob("*.json")]
    return results


@app.local_entrypoint()
def main(
    prediction: str = None,
    predictions_dir: str = None,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    max_workers: int = 16,  # Increased from 4 to 16 for faster parallel evaluation
    list_only: bool = False,
    download: str = None,
):
    """
    Main entry point for Modal evaluation.
    
    Args:
        prediction: Path to single prediction file to evaluate
        predictions_dir: Directory containing prediction files to evaluate
        dataset: SWE-bench dataset to use
        max_workers: Number of parallel workers
        list_only: Only list files in volume, don't evaluate
        download: Download results for prediction name
    """
    import sys
    import os
    from pathlib import Path
    
    # Read Modal credentials from local config
    modal_token_id = None
    modal_token_secret = None
    
    modal_config_path = Path.home() / ".modal.toml"
    if modal_config_path.exists():
        print("🔑 Reading Modal credentials from ~/.modal.toml")
        import toml
        try:
            config = toml.load(modal_config_path)
            # Find the active profile
            for profile_name, profile_data in config.items():
                if isinstance(profile_data, dict) and profile_data.get("active"):
                    modal_token_id = profile_data.get("token_id")
                    modal_token_secret = profile_data.get("token_secret")
                    print(f"✅ Found active Modal profile: {profile_name}")
                    break
            
            if not modal_token_id:
                print("⚠️  Warning: No active Modal profile found")
        except Exception as e:
            print(f"⚠️  Warning: Could not read Modal credentials: {e}")
    else:
        print("⚠️  Warning: ~/.modal.toml not found. Run 'modal token new' first.")
    
    # Handle list command
    if list_only:
        print("📋 Predictions in volume:")
        predictions = list_predictions.remote()
        for p in predictions:
            print(f"  - {p}")
        
        print("\n📊 Results in volume:")
        results = list_results.remote()
        for r in results:
            print(f"  - {r}")
        
        return
    
    # Handle download command
    if download:
        print(f"📥 Downloading results for: {download}")
        results = download_results.remote(download)
        
        if "error" in results:
            print(f"❌ {results['error']}")
            sys.exit(1)
        
        print(f"\n✅ Results:")
        print(f"Status: {results['status']}")
        if results['status'] == 'success':
            print(f"\n{results['stdout']}")
        else:
            print(f"\nError: {results.get('error', 'Unknown error')}")
        
        return
    
    # Collect prediction files
    prediction_files = []
    
    if prediction:
        pred_path = Path(prediction)
        if not pred_path.exists():
            print(f"❌ Prediction file not found: {prediction}")
            sys.exit(1)
        prediction_files.append(pred_path)
    
    elif predictions_dir:
        pred_dir = Path(predictions_dir)
        if not pred_dir.exists():
            print(f"❌ Predictions directory not found: {predictions_dir}")
            sys.exit(1)
        
        # Find all SWE-bench prediction files (both naming patterns)
        prediction_files = list(pred_dir.glob("swebench*.jsonl")) + list(pred_dir.glob("swe_bench*.jsonl"))
        
        if not prediction_files:
            print(f"❌ No SWE-bench prediction files found in: {predictions_dir}")
            print(f"   Looking for files matching: swebench*.jsonl or swe_bench*.jsonl")
            sys.exit(1)
    
    else:
        print("❌ Must specify --prediction or --predictions-dir")
        sys.exit(1)
    
    print(f"🚀 Starting SWE-bench evaluation on Modal")
    print(f"📦 Found {len(prediction_files)} prediction file(s)")
    print(f"📊 Dataset: {dataset}")
    print(f"⚙️  Workers: {max_workers}")
    print()
    
    # Process each prediction file
    results = []
    for i, pred_file in enumerate(prediction_files, 1):
        print(f"[{i}/{len(prediction_files)}] Processing: {pred_file.name}")
        
        # Read file content locally
        with open(pred_file, 'r') as f:
            content = f.read()
        
        # Upload to Modal
        volume_path = upload_prediction.remote(pred_file.name, content)
        
        # Run evaluation
        try:
            result = evaluate_prediction.remote(
                volume_path,
                dataset=dataset,
                max_workers=max_workers,
                modal_token_id=modal_token_id,
                modal_token_secret=modal_token_secret,
            )
            results.append(result)
            print(f"✅ Completed: {pred_file.name}\n")
        
        except Exception as e:
            print(f"❌ Failed: {pred_file.name}")
            print(f"   Error: {e}\n")
            results.append({
                "prediction_file": str(pred_file),
                "status": "failed",
                "error": str(e),
            })
    
    # Summary
    print("=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in results if r.get("status") == "success")
    failed = len(results) - successful
    
    print(f"Total evaluations: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print()
    
    # Download and save results locally
    print("💾 Saving results locally...")
    local_results_dir = Path("results")
    local_results_dir.mkdir(exist_ok=True)
    
    for result in results:
        if result.get("status") == "success":
            pred_name = Path(result["prediction_file"]).stem
            
            # Download from Modal
            modal_results = download_results.remote(pred_name)
            
            # Save locally
            local_path = local_results_dir / f"{pred_name}_results.json"
            import json
            with open(local_path, 'w') as f:
                json.dump(modal_results, f, indent=2)
            
            print(f"   Saved: {local_path}")
    
    print()
    print("✨ Done! Results saved to results/ directory")
