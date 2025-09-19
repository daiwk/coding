## Structure of the repo
`embedding_adapter_prompt`: Model built on public praxis and paxml libraries
`interleaved_transformer_lm`: Model built on public praxis and paxml libraries
`feature_converters`: Model built on public praxis and paxml libraries
`cluster`: Preprocessing script - supports kmeans, random, frequency.
`metrics`: Evaluation metrics - NDCG and recall
`utils`: Utils for interleaving item token and text token

The code is built on top of open source Paxml

To run the model and setup cloud TPUs refer to instructions at
https://github.com/google/paxml/tree/main