import os
import argparse
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify

from src.utils import load_config, get_device, get_vram_info, setup_logger
from src.model import DefectXEngine


class DefectXService:
    """
    Programmatic Python API for DefectX Engine.
    Used by Python services, test harnesses, or teammates.
    """
    def __init__(
        self,
        config_path: str = "config/default_config.yaml",
        checkpoint_path: str = "checkpoints/defectx_bottle.pt",
        device_str: str = "cuda",
    ):
        self.config = load_config(config_path)
        self.device = get_device(prefer_cuda=(device_str.lower() != "cpu"))
        self.engine = DefectXEngine(config=self.config, device=self.device)
        if os.path.exists(checkpoint_path):
            self.engine.load_checkpoint(checkpoint_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'. Run train.py first.")

    def predict(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Runs inference on a single image and returns JSON-compatible dict."""
        return self.engine.predict(image_path=image_path, output_dir=output_dir)


def create_app(service: DefectXService) -> Flask:
    """Creates Flask HTTP microservice for PHP backend integration."""
    app = Flask("DefectX-AI-Engine")
    logger = setup_logger("defectx")

    @app.route("/health", methods=["GET"])
    def health():
        vram = get_vram_info()
        return jsonify({
            "status": "online",
            "model_version": service.engine.version,
            "device": str(service.device),
            "vram": vram,
        }), 200

    @app.route("/predict", methods=["POST"])
    def predict_endpoint():
        try:
            # Check if JSON payload with image_path was sent
            data = request.get_json(silent=True)
            if data and "image_path" in data:
                img_path = data["image_path"]
            elif "image" in request.files:
                # Handle direct file upload
                file = request.files["image"]
                upload_dir = "results/uploads"
                os.makedirs(upload_dir, exist_ok=True)
                img_path = os.path.join(upload_dir, file.filename)
                file.save(img_path)
            else:
                return jsonify({
                    "error": "Request must include JSON {'image_path': '...'} or multipart form file 'image'",
                    "prediction": None,
                }), 400

            result = service.predict(img_path)
            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Prediction API error: {str(e)}")
            return jsonify({
                "error": str(e),
                "prediction": None,
                "anomaly_score": None,
                "severity": None,
                "confidence": None,
                "heatmap_path": None,
                "model_version": service.engine.version,
                "processing_time_ms": 0.0,
                "shift_detected": None,
            }), 500

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start DefectX AI HTTP microservice for PHP backend.")
    parser.add_argument("--config", type=str, default="config/default_config.yaml")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/defectx_bottle.pt")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    service = DefectXService(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device_str=args.device,
    )
    app = create_app(service)
    print(f"DefectX AI HTTP Service running at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
