use mistralrs::{IsqType, Model, TextModelBuilder};

pub async fn load_text_model(model_name: String) -> Result<Model, anyhow::Error> {
    let model = TextModelBuilder::new(model_name)
        .with_isq(IsqType::Q4K)
        .build()
        .await?;

    Ok(model)
}