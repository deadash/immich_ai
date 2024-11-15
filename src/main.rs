use actix_multipart::form::bytes::Bytes;
use actix_multipart::form::text::Text;
use actix_multipart::form::MultipartForm;
use actix_web::{web, App, HttpServer, HttpResponse};
use serde::Deserialize;
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH, Duration};
use log::{info, error, debug,};
use actix_web::middleware::Logger;
use crate::models::SafeModelManager;

mod models;
mod model_registry;

#[derive(Debug, Deserialize)]
struct ModelConfig {
    #[serde(rename = "modelName")]
    model_name: String,
    #[serde(default)]
    options: Value,
}

#[derive(Debug, Deserialize)]
struct ClipContent {
    #[serde(default)]
    textual: Option<ModelConfig>,
    #[serde(default)]
    visual: Option<ModelConfig>,
}

#[derive(Debug, Deserialize)]
struct ClipRequest {
    clip: ClipContent,
}

#[derive(Debug, MultipartForm)]
struct PredictForm {
    entries: Text<String>,
    text: Option<Text<String>>,
    image: Option<Bytes>,
}

struct AppState {
    model_manager: SafeModelManager,
}

async fn predict(
    state: web::Data<AppState>,
    MultipartForm(form): MultipartForm<PredictForm>,
) -> Result<HttpResponse, actix_web::Error> {
    let entries: ClipRequest = serde_json::from_str(&form.entries.into_inner())?;

    // 处理图像模型
    if let Some(visual_config) = &entries.clip.visual {
        let model = state.model_manager.get_model(&visual_config.model_name).await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
        
        if let Some(image_data) = &form.image {
            let output = model.process(&models::ModelInput::Image(image_data.data.to_vec()))
                .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
            return Ok(HttpResponse::Ok().json(json!({
                "clip": output.features
            })));
        }
    }
    
    // 处理文本模型
    if let Some(text_config) = &entries.clip.textual {
        let model = state.model_manager.get_model(
                format!("{}_text", text_config.model_name).as_str()
            ).await
            .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
        
        if let Some(text) = &form.text {
            let output = model.process(&models::ModelInput::Text(text.to_string()))
                .map_err(|e| actix_web::error::ErrorInternalServerError(e))?;
            return Ok(HttpResponse::Ok().json(json!({
                "clip": output.features
            })));
        }
    }
    
    Err(actix_web::error::ErrorBadRequest("No model specified"))
}

// 空闲检查任务
async fn idle_shutdown_task(state: web::Data<AppState>, model_ttl: u64, poll_interval: u64) {
    loop {
        // 清理过期模型
        state.model_manager.cleanup_expired_models(Duration::from_secs(model_ttl)).await;
        tokio::time::sleep(tokio::time::Duration::from_secs(poll_interval)).await;
    }
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    // 设置更详细的日志级别
    std::env::set_var("RUST_LOG", "debug,actix_web=debug");
    env_logger::init();

    // 使用dml或者cuda
    ort::init()
        .with_execution_providers(
            [
                ort::TensorRTExecutionProvider::default().build(),
                ort::CUDAExecutionProvider::default().build(),
                ort::DirectMLExecutionProvider::default().build(),
            ]
        ).commit().expect("Failed to initialize ORT");

    let mut model_manager = SafeModelManager::new();
    model_registry::register_default_models(&mut model_manager).await;

    let state = web::Data::new(AppState {
        model_manager
    });

    info!("Starting server...");

    // 克隆state用于idle任务
    let state_for_idle = state.clone();
    
    // 启动idle检查任务
    let model_ttl = 10 * 60;       // 模型生存时间，单位：秒 (10分钟)
    let poll_interval = 10;        // 检查间隔，单位：秒 (10秒)
    tokio::spawn(async move {
        idle_shutdown_task(state_for_idle, model_ttl, poll_interval).await
    });

    HttpServer::new(move || {
        App::new()
            .wrap(Logger::default()) // 添加默认日志中间件
            .app_data(web::JsonConfig::default().error_handler(|err, _req| {
                error!("JSON 解析错误: {:?}", err);
                error!("错误详情: {}", err.to_string());
                actix_web::error::InternalError::from_response(
                    "",
                    HttpResponse::BadRequest().json(json!({
                        "error": "请求格式错误",
                        "details": err.to_string()
                    }))
                ).into()
            }))
            .app_data(web::FormConfig::default().error_handler(|err, _req| {
                error!("Form 解析错误: {:?}", err);
                error!("错误详情: {}", err.to_string());
                actix_web::error::InternalError::from_response(
                    "",
                    HttpResponse::BadRequest().json(json!({
                        "error": "表单解析错误",
                        "details": err.to_string()
                    }))
                ).into()
            }))
            .app_data(state.clone())
            .route("/", web::get().to(|| async { 
                HttpResponse::Ok().json(json!({"message": "Immich ML"}))
            }))
            .route("/ping", web::get().to(|| async {
                HttpResponse::Ok().body("pong")
            }))
            .route("/predict", web::post().to(predict))
    })
    .bind("0.0.0.0:3003")?
    .run()
    .await
}