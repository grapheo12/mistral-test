// This is ported from https://github.com/huggingface/candle/blob/main/candle-examples/examples/bert/main.rs
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::{backend::BackendDevice, Device, MetalDevice, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::info;
use tokenizers::{PaddingParams, Tokenizer};

pub fn load_model_from_hf(model_name: String) -> Result<(BertModel, Tokenizer, Device), anyhow::Error> {    
    let repo = Repo::new(model_name, RepoType::Model);
    let (config_filename, tokenizer_filename, weights_filename) = {
        let api = Api::new()?;
        let api = api.repo(repo);
        let config = api.get("config.json")?;
        let tokenizer = api.get("tokenizer.json")?;
        let weights = api.get("model.safetensors")?;
        (config, tokenizer, weights)
    };

    let device = Device::Metal(MetalDevice::new(0)?);

    let config = std::fs::read_to_string(config_filename)?;
    let mut config: Config = serde_json::from_str(&config)?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };
    config.hidden_act = HiddenAct::GeluApproximate;
    let model = BertModel::load(vb, &config)?;
    Ok((model, tokenizer, device))
}


/// Returns the sentences and their embeddings.
pub fn compute_sentence_embeddings(embedding_model: &BertModel, tokenizer: &mut Tokenizer, device: &Device, data: String) -> Result<(Vec<String>, Tensor), anyhow::Error> {

        // Split the data into sentences.
        let sentences = data.split(&['.', '?', '!', '\n', '\r', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()).collect::<Vec<_>>();
        let _sentences = sentences.clone();
  
        // Tokenize the sentences.
        let tokenizer = tokenizer
            .with_padding(Some(PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }))
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;

        let tokens = tokenizer.encode_batch(sentences, true)
            .map_err(anyhow::Error::msg)?;

        let token_ids = tokens.iter()
            .map(|t| {
                let tokens = t.get_ids().to_vec();
                Tensor::new(tokens.as_slice(), device).unwrap()
            })
            .collect::<Vec<_>>();
        let attention_mask = tokens.iter()
            .map(|t| {
                let tokens = t.get_attention_mask().to_vec();
                Tensor::new(tokens.as_slice(), device).unwrap()
            })
            .collect::<Vec<_>>();
        
        let token_ids = Tensor::stack(&token_ids, 0)?;
        let attention_mask = Tensor::stack(&attention_mask, 0)?;
        let token_type_ids = Tensor::zeros_like(&token_ids)?;
        
        // Compute the embeddings.
        let embeddings = embedding_model.forward(&token_ids, &token_type_ids, Some(&attention_mask))?;

        let embeddings = normalize_l2(&embeddings)?;
        
        info!("Embeddings: {:?}", embeddings.shape());

        Ok((_sentences, embeddings))

}

fn normalize_l2(v: &Tensor) -> Result<Tensor, anyhow::Error> {
    Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
}

pub struct EmbeddingStore {
    sentences: Vec<String>,
    embeddings: Tensor,
}

impl EmbeddingStore {
    pub fn new(sentences: Vec<String>, embeddings: Tensor) -> Self {
        assert_eq!(sentences.len(), embeddings.shape().dims()[0]);
        Self { sentences, embeddings }
    }
}
