local transformer_model = "roberta-base";
local transformer_dim = 768;
local cls_is_last_token = false;
local max_passage_length = 512;

{
  "dataset_reader":{
    "type": "nela",
    "use_title": "pair",
    "tokenizer": {
      "type": "pretrained_transformer",
      "model_name": transformer_model,
      "max_length": max_passage_length,
      "add_special_tokens": false,
    },
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model,
      }
    }
  },
  "train_data_path": std.extVar("TRAIN_DATA_PATH"),
  "validation_data_path": std.extVar("DEV_DATA_PATH"),
  "test_data_path": std.extVar("TEST_DATA_PATH"),
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model,
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": cls_is_last_token
    },
    "dropout": 0.1,
    "namespace": "tags"
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "batch_size" : 8 
    }
  },
  "trainer": {
    "num_epochs": 10,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}
