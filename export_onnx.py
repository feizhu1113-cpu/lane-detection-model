import torch
from models.lane_classifier import LaneClassifier

model = LaneClassifier()
model.load_state_dict(torch.load("lane_classifier.pth", map_location="cpu"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "lane_classifier.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)

print("✅ 模型已导出为 lane_classifier.onnx")

# 验证
import onnx
onnx_model = onnx.load("lane_classifier.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX 模型验证通过")