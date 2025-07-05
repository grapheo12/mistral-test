// This is ported from https://github.com/huggingface/candle/blob/main/candle-examples/examples/bert/main.rs
use candle_transformers::models::bert::{BertModel, Config, HiddenAct, DTYPE};

use anyhow::{Error as E, Result};
use candle_core::{backend::BackendDevice, Device, MetalDevice, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use log::info;
use tokenizers::{Encoding, PaddingParams, Tokenizer};

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

pub fn get_embedding_from_tokens(tokens: &Vec<Encoding>, embedding_model: &BertModel, device: &Device) -> Result<Tensor, anyhow::Error> {
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

    Ok(embeddings)

}

/// Returns the sentences and their embeddings.
pub fn compute_sentence_embeddings(embedding_model: &BertModel, tokenizer: &mut Tokenizer, device: &Device, padding_params: &Option<PaddingParams>, data: String) -> Result<(Vec<String>, Tensor, Option<PaddingParams>), anyhow::Error> {

        // Split the data into sentences.
        let sentences = data.split(&['.', '?', '!', '\n', '\r', '\t'])
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string()).collect::<Vec<_>>();
        let _sentences = sentences.clone();
  
        let padding_params = match padding_params {
            Some(p) => p.clone(),
            None => PaddingParams {
                strategy: tokenizers::PaddingStrategy::BatchLongest,
                ..Default::default()
            }
        };
        // Tokenize the sentences.
        let tokenizer = tokenizer
            .with_padding(Some(padding_params))
            .with_truncation(None)
            .map_err(anyhow::Error::msg)?;

        let padding_params = tokenizer.get_padding();

        let tokens = tokenizer.encode_batch(sentences, true)
            .map_err(anyhow::Error::msg)?;

        let embeddings = get_embedding_from_tokens(&tokens, embedding_model, device)?;
        Ok((_sentences, embeddings, padding_params.map(|p| p.clone())))

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

    pub fn search(&self, query_embedding: Tensor, k: usize) -> Result<Vec<String>, anyhow::Error> {
        let n = self.sentences.len();
        let mut similarities = Vec::new();
        for i in 0..n {
            let e_i = self.embeddings.get(i)?;
            // The embeddings are normalized, so we can use the dot product to compute the similarity.
            let similarity = (&e_i * &query_embedding)?.sum_all()?.to_scalar::<f32>()?;
            similarities.push((similarity, i));
        }

        similarities.sort_by(|u, v| v.0.total_cmp(&u.0));

        let result = similarities[..k].iter()
            .map(|(_, i)| self.sentences[*i].clone())
            .collect::<Vec<_>>();

        Ok(result)
    }
}

pub fn find_query_embedding(embedding_model: &BertModel, tokenizer: &mut Tokenizer, device: &Device, padding_params: &Option<PaddingParams>, query: String) -> Result<Tensor, anyhow::Error> {
    let padding_params = match padding_params {
        Some(p) => p.clone(),
        None => PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(120),
            ..Default::default()
        }
    };
    let tokenizer = tokenizer.with_padding(Some(padding_params))
        .with_truncation(None)
        .map_err(anyhow::Error::msg)?;
    let tokens = tokenizer.encode_batch(vec![query], true)
        .map_err(anyhow::Error::msg)?;

    let embeddings = get_embedding_from_tokens(&tokens, embedding_model, device)?;
    Ok(embeddings)
}
