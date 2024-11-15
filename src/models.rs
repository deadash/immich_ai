use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use chinese_clip_rs::{ImageProcessor, TextProcessor};
use log::info;
use tokio::sync::Mutex;
use anyhow::{Result, Context};

// 模型特征
pub trait Model: Send + Sync {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput>;
}

// 模型输入枚举
#[derive(Debug)]
pub enum ModelInput {
    Image(Vec<u8>),
    Text(String),
}

// 模型输出
#[derive(Debug, Clone)]
pub struct ModelOutput {
    pub features: Vec<f32>,
}

// 图像处理模型
pub struct ImageModel {
    processor: ImageProcessor,
    image_size: (u32, u32),
}

impl ImageModel {
    pub fn new(model_path: &str, image_size: (u32, u32)) -> Result<Self> {
        Ok(Self {
            processor: ImageProcessor::new(&model_path, image_size)
                .context("Failed to create ImageProcessor")?,
            image_size,
        })
    }
}

impl Model for ImageModel {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Image(image_data) => {
                let features = self.processor.process_image(image_data)?;
                Ok(ModelOutput { features })
            }
            _ => Err(anyhow::anyhow!("Invalid input type for ImageModel")),
        }
    }
}

// 文本处理模型
pub struct TextModel {
    processor: TextProcessor,
    max_length: usize,
}

impl TextModel {
    pub fn new(model_path: &str, tokenizer_path: &str, max_length: usize) -> Result<Self> {
        Ok(Self {
            processor: TextProcessor::new(&model_path, &tokenizer_path, max_length)
                .context("Failed to create TextProcessor")?,
            max_length,
        })
    }
}

impl Model for TextModel {
    fn process(&self, input: &ModelInput) -> Result<ModelOutput> {
        match input {
            ModelInput::Text(text) => {
                let features = self.processor.process_text(text)?;
                Ok(ModelOutput { features })
            }
            _ => Err(anyhow::anyhow!("Invalid input type for TextModel")),
        }
    }
}

// 模型实例包装
struct ModelInstance {
    model: Arc<dyn Model>,
    last_used: Instant,
}

// 模型管理器
pub struct ModelManager {
    models: HashMap<String, ModelInstance>,
    model_configs: HashMap<String, ModelConfig>,
}

#[derive(Clone)]
pub struct ModelConfig {
    pub model_type: ModelType,
    pub model_path: String,
    pub tokenizer_path: Option<String>,
    pub image_size: Option<(u32, u32)>,
    pub max_length: Option<usize>,
}

#[derive(Clone)]
pub enum ModelType {
    Image,
    Text,
}

impl ModelManager {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            model_configs: HashMap::new(),
        }
    }

    pub fn register_model(&mut self, name: String, config: ModelConfig) {
        self.model_configs.insert(name, config);
    }

    pub async fn get_or_load_model(&mut self, name: &str) -> Result<Arc<dyn Model>> {
        if let Some(instance) = self.models.get_mut(name) {
            instance.last_used = Instant::now();
            return Ok(Arc::clone(&instance.model));
        }

        let config = self.model_configs.get(name)
            .ok_or_else(|| anyhow::anyhow!("Model config not found: {}", name))?;

        let model = self.load_model(config).await?;

        self.models.insert(name.to_string(), ModelInstance {
            model: Arc::clone(&model),
            last_used: Instant::now(),
        });

        Ok(model)
    }

    async fn load_model(&self, config: &ModelConfig) -> Result<Arc<dyn Model>> {
        match config.model_type {
            ModelType::Image => {
                let image_size = config.image_size.ok_or_else(|| 
                    anyhow::anyhow!("Image size not specified"))?;
                Ok(Arc::new(ImageModel::new(
                    &config.model_path,
                    image_size,
                )?))
            }
            ModelType::Text => {
                let tokenizer_path = config.tokenizer_path.as_ref()
                    .ok_or_else(|| anyhow::anyhow!("Tokenizer path not specified"))?;
                let max_length = config.max_length
                    .ok_or_else(|| anyhow::anyhow!("Max length not specified"))?;
                Ok(Arc::new(TextModel::new(
                    &config.model_path,
                    &tokenizer_path,
                    max_length,
                )?))
            }
        }
    }

    pub async fn cleanup_expired_models(&mut self, ttl: Duration) {
        let now = Instant::now();
        self.models.retain(|name, instance| {
            let retain = now.duration_since(instance.last_used) < ttl;
            if !retain {
                info!("清理过期模型: {}", name);
            }
            retain
        });
    }

    fn cleanup_oldest_model(&mut self) {
        if let Some((oldest_key, _)) = self.models
            .iter()
            .min_by_key(|(_, instance)| instance.last_used)
            .map(|(k, _)| (k.clone(), ())) {

            info!("清理最旧模型: {}", oldest_key);
            self.models.remove(&oldest_key);
        }
    }
}

// 线程安全的模型管理器包装
pub struct SafeModelManager {
    inner: Arc<Mutex<ModelManager>>,
}

impl SafeModelManager {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(ModelManager::new())),
        }
    }

    pub async fn register_model(&self, name: String, config: ModelConfig) {
        let mut manager = self.inner.lock().await;
        manager.register_model(name, config);
    }

    pub async fn get_model(&self, name: &str) -> Result<Arc<dyn Model>> {
        let mut manager = self.inner.lock().await;
        manager.get_or_load_model(name).await
    }

    pub async fn cleanup_expired_models(&self, ttl: Duration) {
        let mut manager = self.inner.lock().await;
        manager.cleanup_expired_models(ttl).await;
    }
}
