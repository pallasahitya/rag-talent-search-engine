"""Offline helpers for tests: a tiny randomly initialised BERT encoder + WordPiece tokenizer.

It exercises the :class:`~talent_rag.embedders.HFEmbedder` code path (tokenisation, pooling, normalisation) without
internet. Its embeddings carry no real semantics - they only make sure the plumbing is correct.
"""

from __future__ import annotations

from pathlib import Path


def build_tiny_encoder(out_dir: str | Path, texts: list[str], vocab_size: int = 500, hidden_size: int = 32, seed: int = 0) -> Path:
    import torch
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tok.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    cls_id, sep_id = tok.token_to_id("[CLS]"), tok.token_to_id("[SEP]")
    tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)])
    tok.decoder = decoders.WordPiece(prefix="##")
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                   model_input_names=["input_ids", "token_type_ids", "attention_mask"])
    config = BertConfig(vocab_size=len(fast), hidden_size=hidden_size, num_hidden_layers=2, num_attention_heads=2,
                        intermediate_size=hidden_size * 2, max_position_embeddings=512, pad_token_id=fast.pad_token_id)
    torch.manual_seed(seed)
    BertModel(config).save_pretrained(out)
    fast.save_pretrained(out)
    return out
