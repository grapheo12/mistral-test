use mistralrs::{IsqType, Model, TextModelBuilder};

pub async fn load_text_model(model_name: String) -> Result<Model, anyhow::Error> {
    let model = TextModelBuilder::new(model_name)
        .with_isq(IsqType::Q4K)
        .build()
        .await?;

    Ok(model)
}

pub fn transform_query(query: String, results: Vec<String>) -> String {
    format!(
        "Answer the following query:\n {}\n\n
Use the following context to answer the question. Do not use any other information. Always quote the source of the information:\n {}",
        query,
        results.join("\n\n------------------------\n\n")
    )
}