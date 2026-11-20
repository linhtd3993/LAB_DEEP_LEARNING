"""EDA du lieu IMDb va danh gia Output / Error Analysis cho Lab 3."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

LABEL_NAMES = {0: 'NEGATIVE', 1: 'POSITIVE'}

CONTRAST_CUES = (
    'but', 'however', 'although', 'unfortunately', 'though', 'yet', 'despite'
)

# Giu lai cac tinh tu cam xuc (good, bad, great...); chi loai bo stopword trung tinh.
NEUTRAL_STOPWORDS = {
    'the', 'and', 'for', 'that', 'this', 'with', 'was', 'are', 'but', 'not',
    'you', 'have', 'had', 'his', 'her', 'they', 'she', 'him', 'from', 'one',
    'all', 'out', 'about', 'what', 'when', 'which', 'who', 'will', 'would',
    'there', 'their', 'been', 'were', 'has', 'more', 'some', 'can', 'just',
    'like', 'into', 'than', 'then', 'only', 'also', 'very', 'even', 'much',
    'because', 'see', 'get', 'how', 'way', 'make', 'made', 'its', 'other',
    'them', 'could', 'did', 'does', 'doing', 'being', 'over', 'such', 'most',
    'here', 'where', 'why', 'any', 'our', 'your', 'these', 'those', 'two',
    'first', 'one', 'film', 'movie', 'films', 'movies', 'one', 'too', 'now',
    'after', 'before', 'while', 'still', 'back', 'through', 'down', 'off',
    'know', 'think', 'say', 'said', 'seen', 'watch', 'watching', 'watched',
    'people', 'thing', 'things', 'man', 'life', 'years', 'year', 'way',
    'don', 'didn', 'doesn', 'isn', 'wasn', 'aren', 'couldn', 'wouldn', 'won',
}


def clean_review_text(text: str) -> str:
    """Loai bo the HTML <br /> va chuan hoa khoang trang."""
    cleaned = re.sub(r'<br\s*/?>', ' ', text, flags=re.IGNORECASE)
    cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()


def _numeric_stats(values: Sequence[int | float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return {
            'mean': 0.0, 'median': 0.0, 'std': 0.0, 'min': 0.0,
            'p75': 0.0, 'p90': 0.0, 'p95': 0.0, 'max': 0.0,
        }
    return {
        'mean': round(float(np.mean(arr)), 2),
        'median': round(float(np.median(arr)), 2),
        'std': round(float(np.std(arr)), 2),
        'min': round(float(np.min(arr)), 2),
        'p75': round(float(np.percentile(arr, 75)), 2),
        'p90': round(float(np.percentile(arr, 90)), 2),
        'p95': round(float(np.percentile(arr, 95)), 2),
        'max': round(float(np.max(arr)), 2),
    }


def summarize_split_stats(
    texts: Sequence[str],
    labels: Sequence[int],
    token_lengths: Sequence[int],
    max_length: int = 128,
) -> dict[str, Any]:
    """Thong ke EDA cho mot tap du lieu IMDb."""
    n = len(texts)
    if n == 0 or not (len(labels) == len(token_lengths) == n):
        raise ValueError('texts, labels, and token_lengths must have equal nonzero length.')

    word_counts = [len(clean_review_text(t).split()) for t in texts]
    raw_word_counts = [len(t.split()) for t in texts]
    neg_indices = [i for i, y in enumerate(labels) if y == 0]
    pos_indices = [i for i, y in enumerate(labels) if y == 1]

    neg_count = len(neg_indices)
    pos_count = len(pos_indices)
    html_br_count = sum(1 for t in texts if '<br' in t.lower())
    empty_count = sum(1 for t in texts if not clean_review_text(t))
    duplicate_count = n - len(set(texts))

    gt_128 = sum(1 for l in token_lengths if l > 128)
    gt_256 = sum(1 for l in token_lengths if l > 256)
    gt_512 = sum(1 for l in token_lengths if l > 512)
    gt_max = sum(1 for l in token_lengths if l > max_length)

    return {
        'count': n,
        'neg_count': neg_count,
        'pos_count': pos_count,
        'neg_pct': round(100.0 * neg_count / n, 2),
        'pos_pct': round(100.0 * pos_count / n, 2),
        'word_stats': _numeric_stats(word_counts),
        'raw_word_stats': _numeric_stats(raw_word_counts),
        'token_stats': _numeric_stats(token_lengths),
        'by_label': {
            'NEGATIVE': {
                'count': neg_count,
                'word_stats': _numeric_stats([word_counts[i] for i in neg_indices]),
                'token_stats': _numeric_stats([token_lengths[i] for i in neg_indices]),
            },
            'POSITIVE': {
                'count': pos_count,
                'word_stats': _numeric_stats([word_counts[i] for i in pos_indices]),
                'token_stats': _numeric_stats([token_lengths[i] for i in pos_indices]),
            },
        },
        'truncation': {
            'max_length': max_length,
            'gt_max_length_count': gt_max,
            'gt_max_length_pct': round(100.0 * gt_max / n, 2),
            'gt_128_count': gt_128,
            'gt_128_pct': round(100.0 * gt_128 / n, 2),
            'gt_256_count': gt_256,
            'gt_256_pct': round(100.0 * gt_256 / n, 2),
            'gt_512_count': gt_512,
            'gt_512_pct': round(100.0 * gt_512 / n, 2),
        },
        'html_br_count': html_br_count,
        'html_br_pct': round(100.0 * html_br_count / n, 2),
        'empty_count': empty_count,
        'duplicate_count': duplicate_count,
    }


def extract_distinctive_words(
    texts: Sequence[str],
    labels: Sequence[int],
    top_k: int = 12,
    alpha: float = 1.0,
) -> dict[str, list[dict[str, Any]]]:
    """Trich xuat cac tu phan biet manh giua POSITIVE va NEGATIVE bang smoothed log-odds."""
    if len(texts) != len(labels) or not texts:
        raise ValueError('texts and labels must have equal nonzero length.')

    pos_counts: Counter[str] = Counter()
    neg_counts: Counter[str] = Counter()

    for text, label in zip(texts, labels):
        cleaned = clean_review_text(text).lower()
        tokens = re.findall(r'[a-z]{3,}', cleaned)
        filtered = [w for w in tokens if w not in NEUTRAL_STOPWORDS]
        if label == 1:
            pos_counts.update(filtered)
        else:
            neg_counts.update(filtered)

    vocab = set(pos_counts.keys()) | set(neg_counts.keys())
    if not vocab:
        return {'top_positive': [], 'top_negative': []}

    min_total = 10 if len(texts) >= 1000 else (3 if len(texts) >= 100 else 1)
    total_pos = sum(pos_counts.values())
    total_neg = sum(neg_counts.values())
    v_size = len(vocab)

    scored: list[dict[str, Any]] = []
    for word in vocab:
        pc = pos_counts[word]
        nc = neg_counts[word]
        if pc + nc < min_total:
            continue
        p_pos = (pc + alpha) / (total_pos + alpha * v_size)
        p_neg = (nc + alpha) / (total_neg + alpha * v_size)
        log_odds = math.log(p_pos / p_neg)
        scored.append({
            'word': word,
            'score': round(float(log_odds), 4),
            'pos_count': int(pc),
            'neg_count': int(nc),
        })

    top_positive = sorted(scored, key=lambda x: (-x['score'], -x['pos_count'], x['word']))[:top_k]
    top_negative = sorted(scored, key=lambda x: (x['score'], -x['neg_count'], x['word']))[:top_k]
    return {
        'top_positive': top_positive,
        'top_negative': top_negative,
    }


def classify_outcome(true_label: int, pred_label: int) -> str:
    """Phan loai ket qua nhom TP/TN/FP/FN voi 1=POSITIVE la lop duong."""
    if true_label not in (0, 1) or pred_label not in (0, 1):
        raise ValueError('Labels must be 0 or 1.')
    if true_label == 1 and pred_label == 1:
        return 'TP'
    if true_label == 0 and pred_label == 0:
        return 'TN'
    if true_label == 0 and pred_label == 1:
        return 'FP'
    return 'FN'


def _has_contrast_cue(text: str) -> bool:
    lowered = clean_review_text(text).lower()
    pattern = r'\b(' + '|'.join(CONTRAST_CUES) + r')\b'
    return bool(re.search(pattern, lowered))


def build_prediction_records(
    texts: Sequence[str],
    labels: Sequence[int],
    probs: Sequence[Sequence[float]],
    token_lengths: Sequence[int],
    visible_texts: Sequence[str],
    truncated_tails: Sequence[str],
    max_length: int = 128,
) -> list[dict[str, Any]]:
    """Dong goi ban ghi du doan chi tiet cho tung mau test."""
    n = len(texts)
    if not (len(labels) == len(probs) == len(token_lengths) == len(visible_texts) == len(truncated_tails) == n):
        raise ValueError('All input sequences must have equal length.')

    records: list[dict[str, Any]] = []
    for idx in range(n):
        text = texts[idx]
        true_label = int(labels[idx])
        prob_neg = float(probs[idx][0])
        prob_pos = float(probs[idx][1])
        pred_label = 1 if prob_pos >= prob_neg else 0
        confidence = prob_pos if pred_label == 1 else prob_neg
        tok_len = int(token_lengths[idx])
        outcome = classify_outcome(true_label, pred_label)

        records.append({
            'idx': idx,
            'true_label': true_label,
            'true_name': LABEL_NAMES[true_label],
            'pred_label': pred_label,
            'pred_name': LABEL_NAMES[pred_label],
            'outcome': outcome,
            'is_correct': true_label == pred_label,
            'prob_neg': round(prob_neg, 6),
            'prob_pos': round(prob_pos, 6),
            'confidence': round(confidence, 6),
            'word_count': len(clean_review_text(text).split()),
            'token_length': tok_len,
            'truncated': tok_len > max_length,
            'has_html_br': '<br' in text.lower(),
            'has_contrast_cue': _has_contrast_cue(text),
            'visible_text': visible_texts[idx],
            'truncated_tail': truncated_tails[idx],
            'text': text,
        })
    return records


def compute_length_buckets(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nhom ket qua du doan theo cac khoang do dai WordPiece token."""
    bins = [
        ('<=128', lambda l: l <= 128),
        ('129-256', lambda l: 129 <= l <= 256),
        ('257-384', lambda l: 257 <= l <= 384),
        ('>384', lambda l: l > 384),
    ]
    out: list[dict[str, Any]] = []
    for label, cond in bins:
        sub = [r for r in records if cond(r['token_length'])]
        cnt = len(sub)
        correct = sum(1 for r in sub if r['is_correct'])
        fp = sum(1 for r in sub if r['outcome'] == 'FP')
        fn = sum(1 for r in sub if r['outcome'] == 'FN')
        acc = (correct / cnt) if cnt else 0.0
        out.append({
            'bucket': label,
            'count': cnt,
            'correct': correct,
            'accuracy': round(acc, 4),
            'error_rate': round(1.0 - acc if cnt else 0.0, 4),
            'fp': fp,
            'fn': fn,
        })
    return out


def compute_confidence_buckets(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Nhom ket qua du doan theo do tu tin softmax."""
    bins = [
        ('0.50-0.60', 0.50, 0.60, False),
        ('0.60-0.70', 0.60, 0.70, False),
        ('0.70-0.80', 0.70, 0.80, False),
        ('0.80-0.90', 0.80, 0.90, False),
        ('0.90-1.00', 0.90, 1.00, True),
    ]
    out: list[dict[str, Any]] = []
    for label, lo, hi, inclusive_hi in bins:
        if inclusive_hi:
            sub = [r for r in records if lo <= r['confidence'] <= hi]
        else:
            sub = [r for r in records if lo <= r['confidence'] < hi]
        cnt = len(sub)
        correct = sum(1 for r in sub if r['is_correct'])
        acc = (correct / cnt) if cnt else 0.0
        avg_conf = float(np.mean([r['confidence'] for r in sub])) if cnt else 0.0
        out.append({
            'bucket': label,
            'count': cnt,
            'correct': correct,
            'accuracy': round(acc, 4),
            'avg_confidence': round(avg_conf, 4),
        })
    return out


def _group_stats(sub: Sequence[dict[str, Any]]) -> dict[str, Any]:
    cnt = len(sub)
    if cnt == 0:
        return {
            'count': 0, 'mean_confidence': 0.0, 'median_confidence': 0.0,
            'high_conf_85_count': 0, 'mean_token_length': 0.0,
            'median_token_length': 0.0, 'truncated_pct': 0.0,
            'has_html_br_pct': 0.0, 'has_contrast_cue_pct': 0.0,
        }
    confs = [r['confidence'] for r in sub]
    tlens = [r['token_length'] for r in sub]
    return {
        'count': cnt,
        'mean_confidence': round(float(np.mean(confs)), 4),
        'median_confidence': round(float(np.median(confs)), 4),
        'high_conf_85_count': sum(1 for c in confs if c >= 0.85),
        'mean_token_length': round(float(np.mean(tlens)), 2),
        'median_token_length': round(float(np.median(tlens)), 2),
        'truncated_pct': round(100.0 * sum(1 for r in sub if r['truncated']) / cnt, 2),
        'has_html_br_pct': round(100.0 * sum(1 for r in sub if r['has_html_br']) / cnt, 2),
        'has_contrast_cue_pct': round(100.0 * sum(1 for r in sub if r['has_contrast_cue']) / cnt, 2),
    }


def _infer_reason_tag(record: dict[str, Any]) -> str:
    """Gán nhãn nguyên nhân tóm tắt bằng tiếng Việt cho từng ví dụ tiêu biểu."""
    outcome = record['outcome']
    vis_low = record['visible_text'].lower()
    if outcome in ('TP', 'TN'):
        if not record['truncated']:
            return 'Ngắn gọn (<=128 token), từ khóa cảm xúc trực diện ở đầu câu'
        return 'Tín hiệu cảm xúc mạnh và nhất quán ngay trong 128 token đầu'
    if '- - -' in record['visible_text'] or '---' in record['visible_text'] or record['visible_text'].count('/') >= 4:
        return 'Chuỗi ký tự lặp / thẻ HTML chiếm hết cửa sổ 128 token đầu'
    if outcome == 'FP':
        if not record['truncated']:
            return 'Mỉa mai / dùng từ khen theo nghĩa ngược dù không bị cắt token'
        if _has_contrast_cue(record['visible_text']):
            return 'Mở đầu khen / kể bối cảnh, phần chê bị cắt sau 128 token'
        return 'Mất ngữ cảnh kết luận do bị cắt tại giới hạn 128 token'
    if re.search(r"don\s*'?\s*t|not|never|hardly|low rating", vis_low) and record['truncated']:
        return 'Cấu trúc phủ định / khen tương phản và bị cắt phần kết sau 128 token'
    if _has_contrast_cue(record['visible_text']):
        return 'Cảm xúc hỗn hợp / nhượng bộ ở nửa đầu khiến xác suất lệch NEGATIVE'
    return 'Nhiễu ngữ cảnh hoặc bị cắt tại 128 token làm mất nhận xét tích cực'


def _compact_case(record: dict[str, Any]) -> dict[str, Any]:
    return {
        'idx': record['idx'],
        'outcome': record['outcome'],
        'true_name': record['true_name'],
        'pred_name': record['pred_name'],
        'confidence': record['confidence'],
        'prob_neg': record['prob_neg'],
        'prob_pos': record['prob_pos'],
        'token_length': record['token_length'],
        'word_count': record['word_count'],
        'truncated': record['truncated'],
        'has_html_br': record['has_html_br'],
        'has_contrast_cue': record['has_contrast_cue'],
        'reason_summary': _infer_reason_tag(record),
        'visible_snippet': clean_review_text(record['visible_text'])[:320],
        'truncated_tail_snippet': clean_review_text(record['truncated_tail'])[:320],
    }


def summarize_error_analysis(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Tong hop phan tich dung/sai va error analysis tu danh sach ban ghi."""
    n = len(records)
    if n == 0:
        raise ValueError('records must be non-empty.')

    tp_rows = [r for r in records if r['outcome'] == 'TP']
    tn_rows = [r for r in records if r['outcome'] == 'TN']
    fp_rows = [r for r in records if r['outcome'] == 'FP']
    fn_rows = [r for r in records if r['outcome'] == 'FN']

    tp, tn, fp, fn = len(tp_rows), len(tn_rows), len(fp_rows), len(fn_rows)
    acc = (tp + tn) / n
    pos_prec = tp / (tp + fp) if (tp + fp) else 0.0
    pos_rec = tp / (tp + fn) if (tp + fn) else 0.0
    pos_f1 = (2 * pos_prec * pos_rec / (pos_prec + pos_rec)) if (pos_prec + pos_rec) else 0.0

    neg_prec = tn / (tn + fn) if (tn + fn) else 0.0
    neg_rec = tn / (tn + fp) if (tn + fp) else 0.0
    neg_f1 = (2 * neg_prec * neg_rec / (neg_prec + neg_rec)) if (neg_prec + neg_rec) else 0.0

    trunc_rows = [r for r in records if r['truncated']]
    not_trunc_rows = [r for r in records if not r['truncated']]

    def _subset_perf(sub: Sequence[dict[str, Any]]) -> dict[str, Any]:
        c = len(sub)
        corr = sum(1 for r in sub if r['is_correct'])
        return {
            'count': c,
            'correct': corr,
            'errors': c - corr,
            'accuracy': round(corr / c, 4) if c else 0.0,
            'error_rate': round(1.0 - (corr / c), 4) if c else 0.0,
            'fp': sum(1 for r in sub if r['outcome'] == 'FP'),
            'fn': sum(1 for r in sub if r['outcome'] == 'FN'),
        }

    rep_tp = [_compact_case(r) for r in sorted(tp_rows, key=lambda x: -x['confidence'])[:6]]
    rep_tn = [_compact_case(r) for r in sorted(tn_rows, key=lambda x: -x['confidence'])[:6]]
    rep_fp = [_compact_case(r) for r in sorted(fp_rows, key=lambda x: -x['confidence'])[:10]]
    rep_fn = [_compact_case(r) for r in sorted(fn_rows, key=lambda x: -x['confidence'])[:10]]

    return {
        'total_samples': n,
        'correct_count': tp + tn,
        'error_count': fp + fn,
        'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
        'overall': {
            'accuracy': round(acc, 4),
            'error_rate': round(1.0 - acc, 4),
            'macro_f1': round((pos_f1 + neg_f1) / 2.0, 4),
        },
        'per_class': {
            'NEGATIVE': {
                'support': tn + fp,
                'precision': round(neg_prec, 4),
                'recall': round(neg_rec, 4),
                'f1': round(neg_f1, 4),
            },
            'POSITIVE': {
                'support': tp + fn,
                'precision': round(pos_prec, 4),
                'recall': round(pos_rec, 4),
                'f1': round(pos_f1, 4),
            },
        },
        'truncation_impact': {
            'not_truncated': _subset_perf(not_trunc_rows),
            'truncated': _subset_perf(trunc_rows),
        },
        'outcome_groups': {
            'TP': _group_stats(tp_rows),
            'TN': _group_stats(tn_rows),
            'FP': _group_stats(fp_rows),
            'FN': _group_stats(fn_rows),
        },
        'length_buckets': compute_length_buckets(records),
        'confidence_buckets': compute_confidence_buckets(records),
        'representative_cases': {
            'TP': rep_tp,
            'TN': rep_tn,
            'FP': rep_fp,
            'FN': rep_fn,
        },
    }


def plot_eda_class_and_length(
    eda_summary: dict[str, Any],
    train_neg_lens: Sequence[int],
    train_pos_lens: Sequence[int],
    output_path: Path,
) -> None:
    """Ve bieu do phan bo lop va histogram do dai WordPiece token."""
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))

    # Subplot 1: Class balance
    ax = axes[0]
    splits = ['train', 'validation', 'test']
    split_labels = [
        f"Train\n(n={eda_summary['splits']['train']['count']:,})",
        f"Validation\n(n={eda_summary['splits']['validation']['count']:,})",
        f"Test\n(n={eda_summary['splits']['test']['count']:,})",
    ]
    neg_counts = [eda_summary['splits'][s]['neg_count'] for s in splits]
    pos_counts = [eda_summary['splits'][s]['pos_count'] for s in splits]
    neg_pcts = [eda_summary['splits'][s]['neg_pct'] for s in splits]
    pos_pcts = [eda_summary['splits'][s]['pos_pct'] for s in splits]

    x = np.arange(len(splits))
    w = 0.34
    bars_neg = ax.bar(x - w / 2, neg_counts, width=w, color='#d95f5f', label='NEGATIVE (0)')
    bars_pos = ax.bar(x + w / 2, pos_counts, width=w, color='#2b7bba', label='POSITIVE (1)')

    for bar, cnt, pct in zip(bars_neg, neg_counts, neg_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + 22, f'{cnt}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')
    for bar, cnt, pct in zip(bars_pos, pos_counts, pos_pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, cnt + 22, f'{cnt}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(split_labels, fontsize=9.5)
    ax.set_ylabel('Số lượng mẫu (reviews)', fontsize=10)
    ax.set_title('Phân bố nhãn NEGATIVE vs POSITIVE trên 3 tập', fontsize=11.5, fontweight='bold', pad=10)
    ax.set_ylim(0, max(neg_counts + pos_counts) * 1.24)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=True, loc='upper right')

    # Subplot 2: Token length histogram on train
    ax2 = axes[1]
    clip_max = 800
    neg_clip = np.clip(train_neg_lens, 0, clip_max)
    pos_clip = np.clip(train_pos_lens, 0, clip_max)
    bins = np.linspace(0, clip_max, 33)
    ax2.hist(neg_clip, bins=bins, alpha=0.55, color='#d95f5f', edgecolor='white', label='NEGATIVE')
    ax2.hist(pos_clip, bins=bins, alpha=0.50, color='#2b7bba', edgecolor='white', label='POSITIVE')

    tr_trunc = eda_summary['splits']['train']['truncation']
    max_len = tr_trunc['max_length']
    ax2.axvline(max_len, color='#b71c1c', linestyle='--', linewidth=1.8)
    ax2.axvline(256, color='#ef6c00', linestyle=':', linewidth=1.5)
    ax2.axvline(512, color='#4a148c', linestyle='-.', linewidth=1.4)

    ymax = ax2.get_ylim()[1]
    ax2.set_ylim(0, ymax * 1.22)
    top_y = ax2.get_ylim()[1]
    ax2.text(max_len + 8, top_y * 0.91, f"max_length={max_len}\n(cắt {tr_trunc['gt_128_pct']:.1f}% mẫu)",
             color='#b71c1c', fontsize=8.5, fontweight='bold', va='top')
    ax2.text(264, top_y * 0.72, f"256 tok\n({tr_trunc['gt_256_pct']:.1f}% > 256)",
             color='#e65100', fontsize=8, va='top')
    ax2.text(520, top_y * 0.54, f"512 tok (giới hạn BERT)\n({tr_trunc['gt_512_pct']:.1f}% > 512)",
             color='#4a148c', fontsize=8, va='top')

    tr_tok = eda_summary['splits']['train']['token_stats']
    ax2.set_xlabel(
        f'Số WordPiece token (cột cuối gộp mẫu ≥ {clip_max}; max thực tế = {int(tr_tok["max"])})',
        fontsize=9.5,
    )
    ax2.set_ylabel('Số lượng review (Train 2,000)', fontsize=10)
    ax2.set_title('Phân bố độ dài WordPiece token (Train Set)', fontsize=11.5, fontweight='bold', pad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.legend(frameon=True, loc='upper right')

    fig.tight_layout(pad=1.6)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_eda_text_insights(
    eda_summary: dict[str, Any],
    distinctive_words: dict[str, list[dict[str, Any]]],
    output_path: Path,
) -> None:
    """Ve bieu do ty le cat cut / nhieu HTML va top tu phan biet NEG/POS."""
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.5))

    # Left: Truncation rates & HTML noise across splits
    ax = axes[0]
    splits = ['train', 'validation', 'test']
    categories = ['> 128 tok\n(Bị cắt)', '> 256 tok', '> 512 tok', 'Chứa thẻ\n<br /> HTML']
    x = np.arange(len(categories))
    w = 0.27
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    names = ['Train (2,000)', 'Validation (500)', 'Test (500)']

    for idx, (sp, name, col) in enumerate(zip(splits, names, colors)):
        sp_data = eda_summary['splits'][sp]
        vals = [
            sp_data['truncation']['gt_128_pct'],
            sp_data['truncation']['gt_256_pct'],
            sp_data['truncation']['gt_512_pct'],
            sp_data['html_br_pct'],
        ]
        bars = ax.bar(x + (idx - 1) * w, vals, width=w, color=col, label=name)
        for b, v in zip(bars, vals):
            y_text = v + 1.2 if idx != 1 else v + 3.6
            ax.text(b.get_x() + b.get_width() / 2, y_text, f'{v:.1f}%',
                    ha='center', va='bottom', fontsize=7.4, fontweight='bold' if idx == 0 else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9.5)
    ax.set_ylabel('Tỷ lệ (%)', fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_title('Tỷ lệ mất ngữ cảnh (Truncation) & Nhiễu HTML trên 3 tập', fontsize=11.2, fontweight='bold', pad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=True, loc='upper right')

    # Right: Diverging bar chart of distinctive words
    ax2 = axes[1]
    top_neg = distinctive_words['top_negative'][:10]
    top_pos = distinctive_words['top_positive'][:10]
    combined = sorted(top_neg + top_pos, key=lambda d: d['score'])
    words = [d['word'] for d in combined]
    scores = [d['score'] for d in combined]
    bar_colors = ['#d95f5f' if s < 0 else '#2b7bba' for s in scores]

    y = np.arange(len(words))
    ax2.barh(y, scores, color=bar_colors, height=0.72)
    ax2.axvline(0, color='#333333', linewidth=1.0)
    ax2.set_yticks(y)
    ax2.set_yticklabels(words, fontsize=8.8)

    xmin, xmax = min(scores) - 0.55, max(scores) + 0.55
    ax2.set_xlim(xmin, xmax)
    for yi, sc in zip(y, scores):
        offset = 0.06 if sc >= 0 else -0.06
        ha = 'left' if sc >= 0 else 'right'
        ax2.text(sc + offset, yi, f'{sc:+.2f}', va='center', ha=ha, fontsize=7.8)

    ax2.set_xlabel('Smoothed Log-Odds Score (← NEGATIVE | POSITIVE →)', fontsize=9.5)
    ax2.set_title('Top từ phân biệt mạnh nhất trong Train (2,000)', fontsize=11.2, fontweight='bold', pad=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.35)

    fig.tight_layout(pad=1.6)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_error_analysis_overview(
    records: Sequence[dict[str, Any]],
    error_summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Ve 4 subplot tong quan danh gia dung/sai va error analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.2))

    # (0,0) Precision / Recall / F1 by class
    ax0 = axes[0, 0]
    metrics_keys = ['precision', 'recall', 'f1']
    metric_labels = ['Precision', 'Recall', 'F1-Score']
    neg_vals = [error_summary['per_class']['NEGATIVE'][k] * 100 for k in metrics_keys]
    pos_vals = [error_summary['per_class']['POSITIVE'][k] * 100 for k in metrics_keys]
    x = np.arange(len(metrics_keys))
    w = 0.34
    b1 = ax0.bar(x - w / 2, neg_vals, width=w, color='#d95f5f', label='NEGATIVE (n=254)')
    b2 = ax0.bar(x + w / 2, pos_vals, width=w, color='#2b7bba', label='POSITIVE (n=246)')
    for b, v in zip(b1, neg_vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + 1.2, f'{v:.1f}%', ha='center', fontsize=8.8, fontweight='bold')
    for b, v in zip(b2, pos_vals):
        ax0.text(b.get_x() + b.get_width() / 2, v + 1.2, f'{v:.1f}%', ha='center', fontsize=8.8, fontweight='bold')
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_labels, fontsize=10)
    ax0.set_ylim(0, 108)
    ax0.set_ylabel('Tỷ lệ (%)', fontsize=10)
    ax0.set_title('1. Hiệu năng theo từng lớp (Recall NEG thấp hơn rõ rệt)', fontsize=11, fontweight='bold', pad=10)
    ax0.grid(axis='y', linestyle='--', alpha=0.35)
    ax0.legend(frameon=True, loc='lower right')

    # (0,1) Accuracy by token length bucket
    ax1 = axes[0, 1]
    lb = error_summary['length_buckets']
    b_names = [f"{row['bucket']}\n(n={row['count']})" for row in lb]
    b_accs = [row['accuracy'] * 100 for row in lb]
    bar_cols = ['#2e8b57' if i == 0 else '#5b8fb9' for i in range(len(lb))]
    bars = ax1.bar(np.arange(len(lb)), b_accs, width=0.52, color=bar_cols)
    overall_acc = error_summary['overall']['accuracy'] * 100
    ax1.axhline(overall_acc, color='#b71c1c', linestyle='--', linewidth=1.6, label=f'Overall Acc = {overall_acc:.2f}%')
    for b, row in zip(bars, lb):
        v = row['accuracy'] * 100
        ax1.text(b.get_x() + b.get_width() / 2, v + 1.8, f"{v:.1f}%",
                 ha='center', va='bottom', fontsize=9.2, fontweight='bold')
        ax1.text(b.get_x() + b.get_width() / 2, v * 0.52, f"FP={row['fp']}\nFN={row['fn']}",
                 ha='center', va='center', fontsize=8.4, color='white', fontweight='bold')
    ax1.set_xticks(np.arange(len(lb)))
    ax1.set_xticklabels(b_names, fontsize=9)
    ax1.set_ylim(0, 108)
    ax1.set_ylabel('Accuracy (%)', fontsize=10)
    ax1.set_title('2. Accuracy giảm khi review vượt ngưỡng cắt 128 token', fontsize=11, fontweight='bold', pad=10)
    ax1.grid(axis='y', linestyle='--', alpha=0.35)
    ax1.legend(frameon=True, loc='lower left')

    # (1,0) Confidence distribution across TP, TN, FP, FN
    ax2 = axes[1, 0]
    order = ['TP', 'TN', 'FP', 'FN']
    groups = [[r['confidence'] for r in records if r['outcome'] == k] for k in order]
    group_labels = [
        f"{k}\n(n={len(g)})" for k, g in zip(order, groups)
    ]
    bp = ax2.boxplot(groups, positions=np.arange(len(order)), widths=0.48, patch_artist=True,
                     medianprops={'color': '#111111', 'linewidth': 1.8})
    box_colors = ['#66bb6a', '#42a5f5', '#ef5350', '#ffa726']
    for patch, c in zip(bp['boxes'], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.65)
    means = [float(np.mean(g)) if g else 0.0 for g in groups]
    ax2.scatter(np.arange(len(order)), means, color='#000000', marker='D', s=36, zorder=4, label='Mean confidence')
    for i, m in enumerate(means):
        ax2.text(i + 0.27, m, f'{m:.3f}', va='center', fontsize=8.3, fontweight='bold')
    ax2.set_xticks(np.arange(len(order)))
    ax2.set_xticklabels(group_labels, fontsize=9.5)
    ax2.set_ylim(0.45, 1.04)
    ax2.set_ylabel('Độ tự tin dự đoán (Softmax Score)', fontsize=10)
    ax2.set_title('3. Phân bố độ tự tin theo nhóm TP / TN / FP / FN', fontsize=11, fontweight='bold', pad=10)
    ax2.grid(axis='y', linestyle='--', alpha=0.35)
    ax2.legend(frameon=True, loc='lower left')

    # (1,1) Reliability / calibration curve
    ax3 = axes[1, 1]
    cb = error_summary['confidence_buckets']
    xs = [row['avg_confidence'] for row in cb if row['count'] > 0]
    ys = [row['accuracy'] for row in cb if row['count'] > 0]
    counts = [row['count'] for row in cb if row['count'] > 0]
    ax3.plot([0.5, 1.0], [0.5, 1.0], linestyle='--', color='#757575', label='Lý tưởng (y = x)')
    ax3.plot(xs, ys, marker='o', linewidth=2.2, markersize=7, color='#1f77b4', label='Thực tế trên Test (500)')
    for i, (xi, yi, ci) in enumerate(zip(xs, ys, counts)):
        dy = -0.065 if yi > 0.88 else 0.032
        dx = 0.025 if i == 0 else 0.0
        ax3.text(xi + dx, yi + dy, f'n={ci}\n({yi*100:.1f}%)', ha='center', fontsize=8.1)
    ax3.set_xlim(0.48, 1.03)
    ax3.set_ylim(0.30, 1.07)
    ax3.set_xlabel('Độ tự tin trung bình trong khoảng', fontsize=10)
    ax3.set_ylabel('Accuracy thực tế', fontsize=10)
    ax3.set_title('4. Độ chính xác thực tế theo mức độ tự tin (Calibration)', fontsize=11, fontweight='bold', pad=10)
    ax3.grid(True, linestyle='--', alpha=0.35)
    ax3.legend(frameon=True, loc='upper left')

    fig.tight_layout(pad=2.0)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _pick_card_cases(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chon 4 mau dung (2 TP + 2 TN) va 4 mau sai (2 FP + 2 FN) da dang nguyen nhan."""
    tp = sorted([r for r in records if r['outcome'] == 'TP'], key=lambda x: -x['confidence'])
    tn = sorted([r for r in records if r['outcome'] == 'TN'], key=lambda x: -x['confidence'])
    fp = sorted([r for r in records if r['outcome'] == 'FP'], key=lambda x: -x['confidence'])
    fn = sorted([r for r in records if r['outcome'] == 'FN'], key=lambda x: -x['confidence'])

    # Prefer diversity in TP (1 truncated + 1 short if available)
    tp_short = [r for r in tp if not r['truncated']]
    tp_sel = ([tp[0]] if tp else []) + ([tp_short[0]] if tp_short and tp_short[0] != tp[0] else tp[1:2])
    tp_sel = tp_sel[:2]

    tn_sel = tn[:2]

    # For FP: pick highest conf + one non-truncated (sarcasm) or distinct failure mode
    fp_sel: list[dict[str, Any]] = []
    if fp:
        fp_sel.append(fp[0])
        non_trunc_fps = [r for r in fp[1:] if not r['truncated']]
        if non_trunc_fps:
            fp_sel.append(non_trunc_fps[0])
        else:
            contrast_fps = [r for r in fp[1:] if _has_contrast_cue(r['visible_text'])]
            fp_sel.append(contrast_fps[0] if contrast_fps else (fp[1] if len(fp) > 1 else fp[0]))
        fp_sel = fp_sel[:2] if len(fp) >= 2 else fp_sel

    # For FN: pick highest conf + one with noise/spam dashes or HTML noise
    fn_sel: list[dict[str, Any]] = []
    if fn:
        fn_sel.append(fn[0])
        noisy_fns = [
            r for r in fn[1:]
            if '- - -' in r['visible_text'] or '---' in r['visible_text']
        ]
        if noisy_fns:
            fn_sel.append(noisy_fns[0])
        else:
            other_fns = [r for r in fn[1:] if r['idx'] != fn[0]['idx']]
            fn_sel.append(other_fns[0] if other_fns else fn[0])
        fn_sel = fn_sel[:2] if len(fn) >= 2 else fn_sel

    return tp_sel + tn_sel, fp_sel + fn_sel


def plot_correct_vs_wrong_cases(
    records: Sequence[dict[str, Any]],
    output_path: Path,
) -> None:
    """Ve card layout so sanh 4 mau du doan DUNG va 4 mau du doan SAI tieu bieu."""
    correct_cases, wrong_cases = _pick_card_cases(records)
    fig, ax = plt.subplots(figsize=(14.5, 10.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Column headers
    ax.text(0.25, 0.975, 'DỰ ĐOÁN ĐÚNG TIÊU BIỂU (2 TP + 2 TN)',
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#1b5e20')
    ax.text(0.75, 0.975, 'DỰ ĐOÁN SAI TIÊU BIỂU (2 FP + 2 FN)',
            ha='center', va='center', fontsize=12.5, fontweight='bold', color='#b71c1c')

    def draw_column(cases: list[dict[str, Any]], x_left: float, is_correct: bool) -> None:
        card_w = 0.46
        card_h = 0.21
        gap = 0.025
        y_top = 0.94

        for row_idx, rec in enumerate(cases[:4]):
            y0 = y_top - (row_idx + 1) * card_h - row_idx * gap
            if is_correct:
                fc, ec, hdr_col = '#eef8f0', '#2e8b57', '#1b5e20'
            else:
                fc = '#fdf0ef' if rec['outcome'] == 'FP' else '#fff7ed'
                ec = '#c62828' if rec['outcome'] == 'FP' else '#e65100'
                hdr_col = ec

            box = patches.FancyBboxPatch(
                (x_left, y0), card_w, card_h,
                boxstyle='round,pad=0.008,rounding_size=0.012',
                facecolor=fc, edgecolor=ec, linewidth=1.6,
            )
            ax.add_patch(box)

            trunc_str = f"BỊ CẮT ({rec['token_length']} tok > 128)" if rec['truncated'] else f"ĐỦ ({rec['token_length']} tok)"
            header = (
                f"[{rec['outcome']}] Test #{rec['idx']} | Thật: {rec['true_name']} -> "
                f"Đoán: {rec['pred_name']} ({rec['confidence']*100:.1f}%)"
            )
            sub_hdr = f"Độ dài: {trunc_str} | HTML <br>: {'Có' if rec['has_html_br'] else 'Không'}"

            vis_clean = clean_review_text(rec['visible_text'])
            snippet = vis_clean[:165] + ('...' if len(vis_clean) > 165 else '')
            wrapped_snip = textwrap.fill(f'128 tok đầu: "{snippet}"', width=62)
            lines = wrapped_snip.splitlines()[:3]
            wrapped_snip = '\n'.join(lines)

            reason = f"Lý do: {_infer_reason_tag(rec)}"
            wrapped_reason = textwrap.fill(reason, width=62)

            ax.text(x_left + 0.012, y0 + card_h - 0.022, header,
                    fontsize=9.0, fontweight='bold', color=hdr_col, va='top')
            ax.text(x_left + 0.012, y0 + card_h - 0.052, sub_hdr,
                    fontsize=8.0, color='#424242', va='top')
            ax.text(x_left + 0.012, y0 + card_h - 0.082, wrapped_snip,
                    fontsize=8.0, color='#212121', va='top', linespacing=1.28)
            ax.text(x_left + 0.012, y0 + 0.044, wrapped_reason,
                    fontsize=8.1, fontweight='bold', color='#111111', va='top', linespacing=1.2)

    draw_column(correct_cases, 0.02, is_correct=True)
    draw_column(wrong_cases, 0.52, is_correct=False)

    fig.suptitle('So sánh trực quan các trường hợp mô hình phân loại ĐÚNG và SAI trên tập Test (500 mẫu)',
                 fontsize=13.5, fontweight='bold', y=0.995)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.01)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='outputs/finetune/model')
    p.add_argument('--output-dir', default='outputs/finetune')
    p.add_argument('--results-dir', default=None)
    p.add_argument('--train-size', type=int, default=2000)
    p.add_argument('--val-size', type=int, default=500)
    p.add_argument('--test-size', type=int, default=500)
    p.add_argument('--max-length', type=int, default=128)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    if min(args.train_size, args.val_size, args.test_size, args.batch_size) < 1:
        p.error('Sizes and batch-size must be positive.')
    if not 8 <= args.max_length <= 512:
        p.error('max-length must be between 8 and 512.')
    return args


def main(args: argparse.Namespace) -> None:
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_path = Path(args.model)
    if not (model_path / 'config.json').is_file():
        raise FileNotFoundError(
            f'Model not found at {model_path}. Run exercise2_finetune.py first or pass --model.'
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_dataset('stanfordnlp/imdb')
    if args.train_size + args.val_size > len(raw['train']) or args.test_size > len(raw['test']):
        raise ValueError('IMDb allows train-size + val-size <= 25000 and test-size <= 25000.')

    pool = raw['train'].shuffle(seed=args.seed)
    train_ds = pool.select(range(args.train_size))
    val_ds = pool.select(range(args.train_size, args.train_size + args.val_size))
    test_ds = raw['test'].shuffle(seed=args.seed).select(range(args.test_size))

    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    model.to(device).eval()

    print(f'Device: {device} | Model parameters: {sum(p.numel() for p in model.parameters()):,}')

    # 1. EDA on full splits + exact subsets
    def _tok_lens(texts: Iterable[str]) -> list[int]:
        enc = tokenizer(list(texts), truncation=False)
        return [len(ids) for ids in enc['input_ids']]

    full_train_texts = list(raw['train']['text'])
    full_train_labels = list(raw['train']['label'])
    full_test_texts = list(raw['test']['text'])
    full_test_labels = list(raw['test']['label'])

    train_texts = list(train_ds['text'])
    train_labels = list(train_ds['label'])
    val_texts = list(val_ds['text'])
    val_labels = list(val_ds['label'])
    test_texts = list(test_ds['text'])
    test_labels = list(test_ds['label'])

    train_tok_lens = _tok_lens(train_texts)
    val_tok_lens = _tok_lens(val_texts)
    test_tok_lens = _tok_lens(test_texts)

    distinctive_words = extract_distinctive_words(train_texts, train_labels, top_k=12)

    eda_summary = {
        'dataset': 'stanfordnlp/imdb',
        'seed': args.seed,
        'max_length': args.max_length,
        'full_dataset': {
            'train_total': len(full_train_texts),
            'train_neg': sum(1 for y in full_train_labels if y == 0),
            'train_pos': sum(1 for y in full_train_labels if y == 1),
            'test_total': len(full_test_texts),
            'test_neg': sum(1 for y in full_test_labels if y == 0),
            'test_pos': sum(1 for y in full_test_labels if y == 1),
            'unsupervised_total': len(raw['unsupervised']) if 'unsupervised' in raw else 0,
            'train_html_br_pct': round(100.0 * sum(1 for t in full_train_texts if '<br' in t.lower()) / len(full_train_texts), 2),
            'test_html_br_pct': round(100.0 * sum(1 for t in full_test_texts if '<br' in t.lower()) / len(full_test_texts), 2),
            'train_raw_word_stats': _numeric_stats([len(t.split()) for t in full_train_texts]),
            'test_raw_word_stats': _numeric_stats([len(t.split()) for t in full_test_texts]),
        },
        'splits': {
            'train': summarize_split_stats(train_texts, train_labels, train_tok_lens, max_length=args.max_length),
            'validation': summarize_split_stats(val_texts, val_labels, val_tok_lens, max_length=args.max_length),
            'test': summarize_split_stats(test_texts, test_labels, test_tok_lens, max_length=args.max_length),
        },
        'distinctive_words_train': distinctive_words,
    }

    # 2. Inference on held-out test set
    all_probs: list[np.ndarray] = []
    visible_texts: list[str] = []
    truncated_tails: list[str] = []

    with torch.inference_mode():
        for start in range(0, len(test_texts), args.batch_size):
            batch_texts = test_texts[start:start + args.batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors='pt',
            )
            logits = model(**{k: v.to(device) for k, v in enc.items()}).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

    probs_arr = np.vstack(all_probs)

    full_test_enc = tokenizer(test_texts, truncation=False)['input_ids']
    for ids in full_test_enc:
        if len(ids) > args.max_length:
            # DistilBERT them [CLS] o dau va [SEP] o cuoi -> giu lai (max_length - 2) token noi dung.
            vis_ids = ids[:args.max_length - 1]
            tail_ids = ids[args.max_length - 1:]
        else:
            vis_ids = ids
            tail_ids = []
        visible_texts.append(tokenizer.decode(vis_ids, skip_special_tokens=True))
        truncated_tails.append(tokenizer.decode(tail_ids, skip_special_tokens=True) if tail_ids else '')

    records = build_prediction_records(
        texts=test_texts,
        labels=test_labels,
        probs=probs_arr.tolist(),
        token_lengths=test_tok_lens,
        visible_texts=visible_texts,
        truncated_tails=truncated_tails,
        max_length=args.max_length,
    )
    error_summary = summarize_error_analysis(records)

    # 3. Save JSON artifacts
    (out_dir / 'eda_summary.json').write_text(
        json.dumps(eda_summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (out_dir / 'error_analysis.json').write_text(
        json.dumps(error_summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    (out_dir / 'test_predictions.json').write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8'
    )

    # 4. Generate 4 PNG figures
    train_neg_lens = [l for l, y in zip(train_tok_lens, train_labels) if y == 0]
    train_pos_lens = [l for l, y in zip(train_tok_lens, train_labels) if y == 1]

    fig_names = [
        'eda_class_and_length.png',
        'eda_text_insights.png',
        'error_analysis_overview.png',
        'correct_vs_wrong_cases.png',
    ]
    plot_eda_class_and_length(eda_summary, train_neg_lens, train_pos_lens, out_dir / fig_names[0])
    plot_eda_text_insights(eda_summary, distinctive_words, out_dir / fig_names[1])
    plot_error_analysis_overview(records, error_summary, out_dir / fig_names[2])
    plot_correct_vs_wrong_cases(records, out_dir / fig_names[3])

    if args.results_dir:
        res_dir = Path(args.results_dir)
        res_dir.mkdir(parents=True, exist_ok=True)
        for fname in ['eda_summary.json', 'error_analysis.json', *fig_names]:
            shutil.copy2(out_dir / fname, res_dir / fname)

    cm = error_summary['confusion_matrix']
    print(
        f"Test Evaluation -> TN={cm['tn']}, FP={cm['fp']}, FN={cm['fn']}, TP={cm['tp']} | "
        f"Acc={error_summary['overall']['accuracy']:.4f} | "
        f"POS F1={error_summary['per_class']['POSITIVE']['f1']:.4f}"
    )
    print('Saved EDA & Error Analysis artifacts to:', out_dir.resolve())
    if args.results_dir:
        print('Copied lightweight evidence to:', Path(args.results_dir).resolve())


if __name__ == '__main__':
    main(parse_args())
