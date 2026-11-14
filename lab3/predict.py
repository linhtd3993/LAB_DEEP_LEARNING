"""Du doan voi mo hinh da fine-tune va luu tren may."""
import argparse
import json
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='outputs/finetune/model')
    p.add_argument('--text', nargs='+', required=True)
    p.add_argument('--max-length', type=int, default=None)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--output', help='Optional JSON file for prediction results.')
    a = p.parse_args()
    if not (Path(a.model) / 'config.json').is_file():
        p.error('Model not found. Run exercise2_finetune.py first or pass --model.')
    if a.max_length is None:
        training_config = Path(a.model).parent / 'config.json'
        saved = json.loads(training_config.read_text(encoding='utf-8')) if training_config.is_file() else {}
        a.max_length = saved.get('max_length', 128)
    if not isinstance(a.max_length, int) or not 8 <= a.max_length <= 512:
        p.error('max-length must be an integer between 8 and 512.')
    import torch
    from transformers import pipeline
    classifier = pipeline('text-classification', model=a.model, tokenizer=a.model,
                          device=0 if torch.cuda.is_available() and not a.cpu else -1)
    rows = []
    for text, result in zip(a.text, classifier(a.text, truncation=True, max_length=a.max_length)):
        print(f"{text} -> {result['label']} ({result['score']:.4f})")
        rows.append({'text': text, 'label': result['label'], 'score': float(result['score'])})
    if a.output:
        target = Path(a.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')

if __name__ == '__main__':
    main()
