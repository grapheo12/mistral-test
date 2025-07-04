use candle_core::{Device, Tensor};
use candle_transformers::models::bert::BertModel;
use log::info;
use mistralrs::{Model, TextMessageRole, TextMessages};
use tokenizers::{PaddingParams, Tokenizer};

use crate::rag::embedding::EmbeddingStore;
mod embedding;
mod query;

pub struct RAG {
    embedding_model_name: String,
    query_model_name: String,

    embedding_model: Option<(BertModel, Tokenizer, Device)>,
    query_model: Option<Model>,

    embedding_store: Option<EmbeddingStore>,


}


impl RAG {
    pub fn new(embedding_model_name: String, query_model_name: String) -> Self {
        Self {
            embedding_model_name,
            query_model_name,
            embedding_model: None,
            query_model: None,
            embedding_store: None,
        }
    }

    pub async fn init(&mut self) -> Result<(), anyhow::Error> {
        let (embedding_model, tokenizer, device) = embedding::load_model_from_hf(self.embedding_model_name.clone())?;
        self.embedding_model = Some((embedding_model, tokenizer, device));

        let query_model = query::load_text_model(self.query_model_name.clone()).await?;
        self.query_model = Some(query_model);

        Ok(())
    }


    pub async fn load_data(&mut self, data: String) -> Result<(), anyhow::Error> {
        let (embedding_model, tokenizer, device) = self.embedding_model.as_mut().unwrap();
        let (sentences, embeddings) = embedding::compute_sentence_embeddings(embedding_model, tokenizer, device, data)?;
        
        info!("Storing embeddings");
        let embedding_store = EmbeddingStore::new(sentences, embeddings);
        self.embedding_store = Some(embedding_store);
        Ok(())
    }

    pub async fn query(&self, query: String) -> Result<String, anyhow::Error> {
        let query_model = self.query_model.as_ref().unwrap();
        let messages = TextMessages::new().add_message(
            TextMessageRole::User,
            query,
        );
        let response = query_model.send_chat_request(messages).await?;
        Ok(response.choices[0].message.content.as_ref().unwrap().to_string())
    }
}