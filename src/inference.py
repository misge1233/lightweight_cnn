"""
Inference script for ONNX models.
Loads ONNX model and runs inference on a single image.
"""

import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: onnxruntime not installed. Install with: pip install onnxruntime")


def preprocess_image(image_path: str, input_size: int = 224):
    """Preprocess image for inference."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor.numpy()


def run_inference(onnx_path: str, image_path: str, class_names: list = None, 
                 confidence_threshold: float = 0.0):
    """Run inference on a single image using ONNX model.
    
    Args:
        onnx_path: Path to ONNX model file
        image_path: Path to input image
        class_names: List of class names
        confidence_threshold: Minimum confidence threshold for prediction (0.0-1.0)
    """
    if not ONNX_AVAILABLE:
        print("Error: onnxruntime not available")
        return None
    
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Preprocess image
    input_data = preprocess_image(image_path, input_size=224)
    
    # Run inference
    outputs = session.run([output_name], {input_name: input_data})
    logits = outputs[0][0]  # Remove batch dimension
    
    # Apply softmax
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / np.sum(exp_logits)
    
    # Get prediction
    predicted_class_idx = np.argmax(probs)
    predicted_prob = probs[predicted_class_idx]
    
    # Apply confidence thresholding
    if predicted_prob < confidence_threshold:
        predicted_class_name = "UNCERTAIN (below threshold)"
        predicted_class_idx = -1
    else:
        predicted_class_name = class_names[predicted_class_idx] if class_names else f"Class {predicted_class_idx}"
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Image: {image_path}")
    print(f"{'='*60}")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Confidence: {predicted_prob*100:.2f}%")
    if confidence_threshold > 0:
        print(f"Confidence threshold: {confidence_threshold*100:.2f}%")
        if predicted_prob < confidence_threshold:
            print(f"WARNING: Confidence below threshold - prediction may be unreliable")
    print(f"\nAll class probabilities:")
    for i, prob in enumerate(probs):
        class_name = class_names[i] if class_names else f"Class {i}"
        marker = " <-- " if i == np.argmax(probs) else ""
        print(f"  {class_name:20s}: {prob*100:.2f}%{marker}")
    print(f"{'='*60}\n")
    
    return {
        'predicted_class': int(predicted_class_idx) if predicted_class_idx >= 0 else None,
        'predicted_class_name': predicted_class_name if predicted_class_idx >= 0 else "UNCERTAIN",
        'confidence': float(predicted_prob),
        'all_probs': probs.tolist(),
        'above_threshold': predicted_prob >= confidence_threshold if confidence_threshold > 0 else True
    }


def main():
    parser = argparse.ArgumentParser(description='Run inference on image using ONNX model')
    parser.add_argument('--onnx_model', type=str, required=True,
                       help='Path to ONNX model file')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--classes', type=str, nargs='+',
                       default=['fusarium_head_blight', 'healthy', 'septoria', 'stem_rust', 'yellow_rust'],
                       help='Class names (default: wheat disease classes)')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                       help='Minimum confidence threshold (0.0-1.0) for accepting prediction')
    
    args = parser.parse_args()
    
    if not Path(args.onnx_model).exists():
        print(f"Error: ONNX model not found at {args.onnx_model}")
        return
    
    if not Path(args.image).exists():
        print(f"Error: Image not found at {args.image}")
        return
    
    if args.confidence_threshold < 0 or args.confidence_threshold > 1:
        print(f"Error: Confidence threshold must be between 0.0 and 1.0")
        return
    
    run_inference(args.onnx_model, args.image, args.classes, args.confidence_threshold)


if __name__ == '__main__':
    main()

