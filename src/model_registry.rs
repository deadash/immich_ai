use crate::models::{ModelConfig, ModelType, SafeModelManager};
use std::path::PathBuf;

pub async fn register_default_models(model_manager: &mut SafeModelManager) {
    // 定义基础路径
    let models_dir = PathBuf::from("models");
    let tokenizer_path = models_dir.join("clip_cn_tokenizer.json").to_string_lossy().to_string();
    let max_length = Some(52);

    // 注册 ViT-B-16
    model_manager.register_model(
        "ViT-B-16__openai".to_string(),
        ModelConfig {
            model_type: ModelType::Image,
            model_path: models_dir.join("ViT-B-16.img.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: Some((224, 224)),  // ViT-B-16 的标准输入尺寸
            max_length,
        }
    ).await;

    model_manager.register_model(
        "ViT-B-16__openai_text".to_string(),
        ModelConfig {
            model_type: ModelType::Text,
            model_path: models_dir.join("ViT-B-16.txt.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: Some(tokenizer_path.clone()),
            image_size: None,
            max_length,
        }
    ).await;

    // 注册 ViT-L-14
    model_manager.register_model(
        "ViT-L-14__openai".to_string(),
        ModelConfig {
            model_type: ModelType::Image,
            model_path: models_dir.join("ViT-L-14.img.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: Some((224, 224)),
            max_length,
        }
    ).await;

    model_manager.register_model(
        "ViT-L-14__openai_text".to_string(),
        ModelConfig {
            model_type: ModelType::Text,
            model_path: models_dir.join("ViT-L-14.txt.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: Some(tokenizer_path.clone()),
            image_size: None,
            max_length,
        }
    ).await;

    // 注册 ViT-L-14-336
    model_manager.register_model(
        "ViT-L-14-336__openai".to_string(),
        ModelConfig {
            model_type: ModelType::Image,
            model_path: models_dir.join("ViT-L-14-336.img.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: None,
            image_size: Some((336, 336)),  // 注意这个模型使用336x336的输入
            max_length,
        }
    ).await;

    model_manager.register_model(
        "ViT-L-14-336__openai_text".to_string(),
        ModelConfig {
            model_type: ModelType::Text,
            model_path: models_dir.join("ViT-L-14-336.txt.fp32.onnx").to_string_lossy().to_string(),
            tokenizer_path: Some(tokenizer_path.clone()),
            image_size: None,
            max_length,
        }
    ).await;
} 