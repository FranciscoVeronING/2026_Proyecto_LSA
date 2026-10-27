"""Métricas de glosas LSA → español.

Superficie: BLEU (corpus, multi-referencia), ROUGE-L y METEOR (máximo por oración).
Neuronales: BERTScore F1 (es), COMET (glosas como fuente) y BLEURT-20.
Con varias referencias, las neuronales también toman el máximo por oración.
``score_all`` devuelve todo en escala 0–100, más las listas por oración.
"""
from __future__ import annotations

import gc

COMET_MODEL = "Unbabel/wmt22-comet-da"
BLEURT_MODEL = "lucadiliello/BLEURT-20"

_COMET = None
_BLEURT = None


def mean_percent(values: list[float]) -> float:
    if not values:
        return 0.0
    return round(100.0 * sum(values) / len(values), 2)


def score_surface(predictions: list[str], references: list[list[str]]) -> dict:
    """BLEU de corpus y, por oración, el mejor ROUGE-L y METEOR entre referencias."""
    import evaluate

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    meteor_metric = evaluate.load("meteor")
    bleu = bleu_metric.compute(predictions=predictions, references=references)["bleu"]
    rouge_l: list[float] = []
    meteor: list[float] = []
    for prediction, refs in zip(predictions, references):
        rouge_l.append(
            max(rouge_metric.compute(predictions=[prediction], references=[ref])["rougeL"] for ref in refs)
            if refs
            else 0.0
        )
        meteor.append(
            max(meteor_metric.compute(predictions=[prediction], references=[ref])["meteor"] for ref in refs)
            if refs
            else 0.0
        )
    return {"bleu": bleu, "rouge_l": rouge_l, "meteor": meteor}


def score_all(predictions: list[str], references: list[list[str]], sources: list[str]) -> dict:
    """Puntajes de corpus en 0–100 y listas crudas por oración."""
    surface = score_surface(predictions, references)
    neural = score_neural_metrics(predictions, references, sources)
    total = len(predictions) or 1
    return {
        "bleu_score": round(surface["bleu"] * 100, 2),
        "rouge_l_score": round(sum(surface["rouge_l"]) / total * 100, 2),
        "meteor_score": round(sum(surface["meteor"]) / total * 100, 2),
        "bertscore_f1": mean_percent(neural["bertscore_f1"]),
        "comet_score": mean_percent(neural["comet"]),
        "bleurt_score": mean_percent(neural["bleurt"]),
        "rouge_l": surface["rouge_l"],
        "meteor": surface["meteor"],
        "bertscore_raw": neural["bertscore_f1"],
        "comet_raw": neural["comet"],
        "bleurt_raw": neural["bleurt"],
    }


def score_neural_metrics(
    predictions: list[str],
    references: list[list[str]],
    sources: list[str],
) -> dict[str, list[float]]:
    """Devuelve listas por oración: bertscore_f1, comet y bleurt."""
    preds = [_blank(text) for text in predictions]
    refs = [[_blank(ref) for ref in group] or [" "] for group in references]
    srcs = [_blank(text) for text in sources]
    if not preds:
        return {"bertscore_f1": [], "comet": [], "bleurt": []}

    print("Calculando BERTScore, COMET y BLEURT...", flush=True)
    bertscore = _bertscore(preds, refs)
    _empty_cache()
    comet = _max_over_refs(preds, refs, srcs, _comet_scores)
    _empty_cache()
    bleurt = _max_over_refs(preds, refs, srcs, _bleurt_scores)
    _empty_cache()
    return {"bertscore_f1": bertscore, "comet": comet, "bleurt": bleurt}


def _blank(text: str) -> str:
    text = (text or "").strip()
    return text if text else " "


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _empty_cache() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _bertscore(predictions: list[str], references: list[list[str]]) -> list[float]:
    from bert_score import score

    _, _, f1 = score(
        predictions,
        references,
        lang="es",
        verbose=False,
        idf=False,
        rescale_with_baseline=False,
        device=_device(),
        batch_size=16,
        nthreads=2,
    )
    return [float(value) for value in f1]


def _max_over_refs(predictions, references, sources, scorer) -> list[float]:
    best: list[float | None] = [None] * len(predictions)
    width = max(len(group) for group in references)
    for ref_index in range(width):
        selected = [i for i, group in enumerate(references) if ref_index < len(group)]
        scored = scorer(
            [predictions[i] for i in selected],
            [references[i][ref_index] for i in selected],
            [sources[i] for i in selected],
        )
        for index, value in zip(selected, scored):
            if best[index] is None or value > best[index]:
                best[index] = value
    return [0.0 if value is None else float(value) for value in best]


def _patch_comet_encoder() -> None:
    """transformers 5 ya no devuelve (hidden, pooler, layers) como tupla."""
    from comet.encoders.xlmr import XLMREncoder

    if getattr(XLMREncoder.forward, "_lsa_patched", False):
        return

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=kwargs.get("token_type_ids"),
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_states = outputs.last_hidden_state
        all_layers = outputs.hidden_states
        if not all_layers:
            raise RuntimeError("XLM-R no devolvió las capas ocultas que COMET necesita")
        return {
            "sentemb": last_hidden_states[:, 0, :],
            "wordemb": last_hidden_states,
            "all_layers": all_layers,
            "attention_mask": attention_mask,
        }

    forward._lsa_patched = True
    XLMREncoder.forward = forward


def _comet_model():
    global _COMET
    if _COMET is None:
        from comet import download_model, load_from_checkpoint

        _patch_comet_encoder()
        checkpoint = download_model(COMET_MODEL)
        _COMET = load_from_checkpoint(checkpoint)
    return _COMET


def _comet_scores(predictions: list[str], references: list[str], sources: list[str]) -> list[float]:
    import torch

    model = _comet_model()
    samples = [
        {"src": source, "mt": prediction, "ref": reference}
        for prediction, reference, source in zip(predictions, references, sources)
    ]
    use_gpu = torch.cuda.is_available()
    output = model.predict(
        samples,
        batch_size=8,
        gpus=1 if use_gpu else 0,
        progress_bar=False,
        accelerator="gpu" if use_gpu else "cpu",
        num_workers=0,
    )
    return [float(value) for value in output.scores]


def _patch_bleurt_import() -> None:
    """transformers 5 movió APIs que bleurt-pytorch importa al cargar."""
    import sys
    import types

    import transformers.pytorch_utils as pytorch_utils

    if not hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):

        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            import torch

            heads = set(heads) - set(already_pruned_heads)
            mask = torch.ones(n_heads, head_size)
            heads_to_prune = set()
            for head in heads:
                adjusted = head - sum(1 for pruned in already_pruned_heads if pruned < head)
                heads_to_prune.add(adjusted)
                mask[adjusted] = 0
            mask = mask.view(-1).eq(1)
            index = torch.arange(len(mask))[mask].long()
            return heads_to_prune, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    module_name = "transformers.models.bert.tokenization_bert_fast"
    if module_name in sys.modules:
        return
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        stub = types.ModuleType(module_name)

        class BertTokenizerFast:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise OSError("Este BLEURT no usa el tokenizer WordPiece.")

        stub.BertTokenizerFast = BertTokenizerFast
        sys.modules[module_name] = stub


def _bleurt_bundle():
    global _BLEURT
    if _BLEURT is None:
        import sentencepiece as spm
        import torch
        from huggingface_hub import hf_hub_download

        _patch_bleurt_import()
        from bleurt_pytorch import BleurtForSequenceClassification
        from bleurt_pytorch.bleurt.modeling_bleurt import BleurtModel

        if not hasattr(BleurtModel, "get_head_mask"):

            def get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
                del self, is_attention_chunked
                if head_mask is None:
                    return [None] * num_hidden_layers
                return head_mask

            BleurtModel.get_head_mask = get_head_mask

        sp = spm.SentencePieceProcessor()
        sp.Load(hf_hub_download(BLEURT_MODEL, "spm.model"))
        special = {
            name: sp.piece_to_id(piece)
            for name, piece in (("cls", "[CLS]"), ("sep", "[SEP]"), ("pad", "[PAD]"))
        }
        if any(token_id < 0 for token_id in special.values()):
            raise RuntimeError("El SentencePiece de BLEURT-20 no tiene [CLS], [SEP] o [PAD]")

        model = BleurtForSequenceClassification.from_pretrained(BLEURT_MODEL)
        model.eval()
        device = _device()
        model.to(device)
        _BLEURT = (sp, special, model, device)
    return _BLEURT


def _bleurt_pair_ids(sp, special, reference: str, prediction: str) -> tuple[list[int], list[int]]:
    ref_ids = sp.encode(reference, out_type=int)
    pred_ids = sp.encode(prediction, out_type=int)
    ids = [special["cls"], *ref_ids, special["sep"], *pred_ids, special["sep"]]
    types = [0] * (len(ref_ids) + 2) + [1] * (len(pred_ids) + 1)
    return ids, types


def _bleurt_scores(predictions: list[str], references: list[str], sources: list[str]) -> list[float]:
    del sources
    import torch

    sp, special, model, device = _bleurt_bundle()
    scores: list[float] = []
    batch_size = 8
    for start in range(0, len(predictions), batch_size):
        pairs = [
            _bleurt_pair_ids(sp, special, reference, prediction)
            for reference, prediction in zip(
                references[start : start + batch_size],
                predictions[start : start + batch_size],
            )
        ]
        width = max(len(ids) for ids, _types in pairs)
        input_ids = []
        token_types = []
        attention = []
        for ids, types in pairs:
            pad = width - len(ids)
            input_ids.append(ids + [special["pad"]] * pad)
            token_types.append(types + [0] * pad)
            attention.append([1] * len(ids) + [0] * pad)
        encoded = {
            "input_ids": torch.tensor(input_ids, device=device),
            "token_type_ids": torch.tensor(token_types, device=device),
            "attention_mask": torch.tensor(attention, device=device),
        }
        with torch.no_grad():
            logits = model(**encoded).logits.flatten()
        scores.extend(float(value) for value in logits.detach().cpu())
    return scores
