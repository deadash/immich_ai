# Immich 机器学习识别服务

这是一个为 Immich 照片管理系统开发的独立机器学习服务，基于 Chinese-CLIP 模型，用于实现图片识别和中文文本理解功能。

## 主要特性

- 支持图片-文本跨模态理解
- 支持中文文本处理
- 支持多种硬件加速方案:
  - CUDA (NVIDIA GPU)
  - DirectX 12 
  - TensorRT
  - CPU

## 支持的模型

目前支持以下预训练模型:

### ViT-B-16 (Immich模型名称: ViT-B-16__openai)
- 图像编码器: [ViT-B-16.img.fp32.onnx](https://www.modelscope.cn/models/deadash/chinese_clip_ViT-B-16/resolve/master/models/Vit-B-16.img.fp32.onnx)
- 文本编码器: [ViT-B-16.txt.fp32.onnx](https://www.modelscope.cn/models/deadash/chinese_clip_ViT-B-16/resolve/master/models/Vit-B-16.txt.fp32.onnx)
- 输入图像尺寸: 224x224

### ViT-L-14 (Immich模型名称: ViT-L-14__openai)
- 图像编码器: [ViT-L-14.img.fp32.onnx](https://www.modelscope.cn/models/deadash/chinese_clip_ViT-L-14/resolve/master/models/Vit-L-14.img.fp32.onnx)
- 文本编码器: [ViT-L-14.txt.fp32.onnx](https://www.modelscope.cn/models/deadash/chinese_clip_ViT-L-14/resolve/master/models/Vit-L-14.txt.fp32.onnx)
- 输入图像尺寸: 224x224

### ViT-L-14-336 (Immich模型名称: ViT-L-14-336__openai)
- 图像编码器: [ViT-L-14-336.img.fp32.onnx](https://www.modelscope.cn/models/deadash/chinese_clip_ViT-L-14-336/resolve/master/models/Vit-L-14-336.img.fp32.onnx)
- 文本编码器: [ViT-L-14-336.txt.fp32.onnx](https://www.modelscope.cn/models/deadash/chinese_clip_ViT-L-14-336/resolve/master/models/Vit-L-14-336.txt.fp32.onnx)
- 输入图像尺寸: 336x336

### 分词器
所有文本模型共用同一个分词器:
- [clip_cn_tokenizer.json](https://www.modelscope.cn/models/deadash/chinese_clip/resolve/master/clip_cn_tokenizer.json)

## 使用说明

1. 下载所需模型文件到 `models` 目录
2. 确保分词器文件 `clip_cn_tokenizer.json` 位于 `models` 目录
3. 程序会自动注册所有可用的模型
4. 可以通过配置选择合适的硬件加速方案

## 性能建议

- ViT-B-16: 适合普通场景，性能和精度平衡
- ViT-L-14: 更高的精度，需要更多计算资源
- ViT-L-14-336: 支持更大的输入图像，适合需要更细节识别的场景