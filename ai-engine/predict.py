import os
import sys
import json
import argparse
from typing import Dict, Any, Optional
import torch

from src.utils import setup_logger, load_config, get_device, get_vram_info
from src.dataset_adapter import validate_image_file, VALID_IMAGE_EXTENSIONS
from src.model import DefectXEngine


# Module-level engine cache to avoid reloading weights in long-running Python processes
_CACHED_ENGINE: Optional[DefectXEngine] = None
_CACHED_CHECKPOINT_PATH: Optional[str] = None


def get_engine(
    checkpoint_path: str = "checkpoints/defectx_bottle.pt",
    config_path: str = "config/default_config.yaml",
    device_str: Optional[str] = None,
) -> DefectXEngine:
    """
    Instantiates or retrieves cached DefectXEngine instance for fast inference.
    """
    global _CACHED_ENGINE, _CACHED_CHECKPOINT_PATH

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model checkpoint not found: '{checkpoint_path}'")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: '{config_path}'")

    if _CACHED_ENGINE is not None and _CACHED_CHECKPOINT_PATH == checkpoint_path:
        return _CACHED_ENGINE

    config = load_config(config_path)
    prefer_cuda = (device_str.lower() != "cpu") if device_str else True
    device = get_device(prefer_cuda=prefer_cuda, verbose=False)

    engine = DefectXEngine(config=config, device=device)
    engine.load_checkpoint(checkpoint_path)

    _CACHED_ENGINE = engine
    _CACHED_CHECKPOINT_PATH = checkpoint_path
    return engine


def predict(
    image_path: str,
    checkpoint_path: str = "checkpoints/defectx_bottle.pt",
    config_path: str = "config/default_config.yaml",
    device_str: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_overlay: bool = True,
    inspection_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Stable Python inference function.
    Validates image path and format, executes nearest-neighbor anomaly detection,
    and returns consistent JSON-compatible dictionary.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: '{image_path}'")

    is_valid, err_msg, _ = validate_image_file(image_path)
    if not is_valid:
        raise ValueError(err_msg or f"Invalid image file: '{image_path}'")

    engine = get_engine(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        device_str=device_str,
    )

    return engine.predict(
        image_path=image_path,
        output_dir=output_dir,
        save_overlay=save_overlay,
        inspection_id=inspection_id,
    )


def check_health(
    checkpoint_path: str = "checkpoints/defectx_bottle.pt",
    config_path: str = "config/default_config.yaml",
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Health check utility:
    Reports checkpoint availability, model loadability, and active inference device.
    """
    checkpoint_exists = os.path.exists(checkpoint_path)
    config_exists = os.path.exists(config_path)

    vram = get_vram_info()
    prefer_cuda = (device_str.lower() != "cpu") if device_str else True
    device = get_device(prefer_cuda=prefer_cuda, verbose=False)

    if not checkpoint_exists:
        return {
            "status": "unhealthy",
            "checkpoint_exists": False,
            "checkpoint_path": os.path.abspath(checkpoint_path) if checkpoint_path else None,
            "config_exists": config_exists,
            "model_loadable": False,
            "model_version": None,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "vram": vram,
            "error": f"Checkpoint not found at '{checkpoint_path}'",
        }

    try:
        engine = get_engine(checkpoint_path=checkpoint_path, config_path=config_path, device_str=device_str)
        return {
            "status": "healthy",
            "checkpoint_exists": True,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "config_exists": config_exists,
            "model_loadable": True,
            "model_version": engine.version,
            "device": str(engine.device),
            "cuda_available": torch.cuda.is_available(),
            "vram": vram,
            "image_threshold": round(engine.threshold_selector.image_threshold, 4),
            "pixel_threshold": round(engine.threshold_selector.pixel_threshold, 4),
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "checkpoint_exists": True,
            "checkpoint_path": os.path.abspath(checkpoint_path),
            "config_exists": config_exists,
            "model_loadable": False,
            "model_version": None,
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "vram": vram,
            "error": f"Failed to load model checkpoint: {str(e)}",
        }


def format_error_response(error_message: str, error_code: str) -> Dict[str, Any]:
    """Standardized error dictionary matching PHP integration contract."""
    return {
        "error": error_message,
        "error_code": error_code,
        "prediction": None,
        "anomaly_score": None,
        "severity": None,
        "confidence": None,
        "heatmap_path": None,
        "overlay_path": None,
        "model_version": None,
        "processing_time_ms": 0.0,
        "shift_detected": None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="DefectX AI Inference Engine - PHP Backend Integration CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image", type=str, default=None, help="Path to input image to inspect")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/defectx_bottle.pt", help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/default_config.yaml", help="Path to configuration file")
    parser.add_argument("--output-dir", type=str, default="results/heatmaps", help="Directory where heatmaps/overlays are saved")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Compute device")
    parser.add_argument("--no-overlay", action="store_true", help="Disable blended overlay generation")
    parser.add_argument("--inspection-id", type=str, default=None, help="Optional external unique inspection ID")
    parser.add_argument("--health", action="store_true", help="Perform model health check and report device status")
    args = parser.parse_args()

    # Logger outputs strictly to stderr so stdout remains clean JSON
    logger = setup_logger("defectx")

    # 1. Health check branch
    if args.health:
        health_info = check_health(
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device_str=args.device,
        )
        print(json.dumps(health_info, indent=2))
        sys.exit(0 if health_info["status"] == "healthy" else 1)

    # 2. Image prediction branch
    if not args.image:
        err = format_error_response("Missing required argument: --image <path>", "MISSING_ARGUMENT")
        sys.stderr.write("ERROR: Missing --image argument\n")
        print(json.dumps(err, indent=2))
        sys.exit(1)

    # Check 1: Missing image
    if not os.path.exists(args.image):
        err = format_error_response(f"Input image not found: '{args.image}'", "MISSING_IMAGE")
        sys.stderr.write(f"ERROR: File not found: '{args.image}'\n")
        print(json.dumps(err, indent=2))
        sys.exit(1)

    # Check 2: Missing checkpoint
    if not os.path.exists(args.checkpoint):
        err = format_error_response(f"Model checkpoint not found: '{args.checkpoint}'", "MISSING_CHECKPOINT")
        sys.stderr.write(f"ERROR: Checkpoint not found: '{args.checkpoint}'\n")
        print(json.dumps(err, indent=2))
        sys.exit(2)

    # Check 3: Unsupported format or corrupted image
    is_valid, validation_err, _ = validate_image_file(args.image)
    if not is_valid:
        err = format_error_response(validation_err or f"Invalid image format: '{args.image}'", "INVALID_IMAGE")
        sys.stderr.write(f"ERROR: {validation_err}\n")
        print(json.dumps(err, indent=2))
        sys.exit(3)

    # Check 4: Inference execution
    try:
        result = predict(
            image_path=args.image,
            checkpoint_path=args.checkpoint,
            config_path=args.config,
            device_str=args.device,
            output_dir=args.output_dir,
            save_overlay=(not args.no_overlay),
            inspection_id=args.inspection_id,
        )
        # Output clean JSON to stdout
        print(json.dumps(result, indent=2))
        sys.exit(0)

    except Exception as e:
        err = format_error_response(f"Inference execution failed: {str(e)}", "INFERENCE_ERROR")
        sys.stderr.write(f"ERROR: Inference failure: {str(e)}\n")
        print(json.dumps(err, indent=2))
        sys.exit(4)


if __name__ == "__main__":
    main()
