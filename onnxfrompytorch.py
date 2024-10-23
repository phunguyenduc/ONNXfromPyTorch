import torch
import onnxruntime
from ultralytics import YOLO

def export_to_onnx(pt_model_path, onnx_model_path, dummy_input_shape):

    model = YOLO(pt_model_path)
    
    # Set the model to evaluation mode
    # model.eval()

    # Provide a sample input (dummy input) in the expected shape
    dummy_input = torch.randn(*dummy_input_shape)

    # Export the model to ONNX format
    torch.onnx.export(model.model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"])

    print(f"Model exported to: {onnx_model_path}")

    # Verify exported model by loading and printing model inputs and outputs
    session = onnxruntime.InferenceSession(onnx_model_path)
    print("Model inputs:", session.get_inputs())
    print("Model outputs:", session.get_outputs())

if __name__ == "__main__":
    # Paths to input and output files
    pt_model_path = "best.pt"  # Path to your YOLO PyTorch model file
    onnx_model_path = "model.onnx"  # Path to save the exported ONNX model

    # Example dummy input shape (batch_size, channels, height, width)
    dummy_input_shape = (1, 3, 640, 640)  # Use the input size expected by the model (e.g., 640x640 for YOLOv8)

    # Export the model to ONNX format
    export_to_onnx(pt_model_path, onnx_model_path, dummy_input_shape)
