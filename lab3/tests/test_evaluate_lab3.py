import json
import unittest
from pathlib import Path

from evaluate_lab3 import (
    build_prediction_records,
    classify_outcome,
    clean_review_text,
    compute_confidence_buckets,
    compute_length_buckets,
    extract_distinctive_words,
    summarize_error_analysis,
    summarize_split_stats,
)


class EvaluateLab3UnitTests(unittest.TestCase):
    def test_clean_review_text_strips_html_and_normalizes_spaces(self):
        raw = 'Great film!<br /><br/>  Worth watching   twice.<br> '
        self.assertEqual(clean_review_text(raw), 'Great film! Worth watching twice.')

    def test_classify_outcome_all_cases(self):
        self.assertEqual(classify_outcome(1, 1), 'TP')
        self.assertEqual(classify_outcome(0, 0), 'TN')
        self.assertEqual(classify_outcome(0, 1), 'FP')
        self.assertEqual(classify_outcome(1, 0), 'FN')
        with self.assertRaises(ValueError):
            classify_outcome(2, 0)

    def test_summarize_split_stats(self):
        texts = [
            'Amazing movie<br />loved it',
            'Terrible boring waste of time',
            'Terrible boring waste of time',
            'Good acting overall',
        ]
        labels = [1, 0, 0, 1]
        token_lengths = [10, 150, 270, 520]
        stats = summarize_split_stats(texts, labels, token_lengths, max_length=128)
        self.assertEqual(stats['count'], 4)
        self.assertEqual(stats['neg_count'], 2)
        self.assertEqual(stats['pos_count'], 2)
        self.assertEqual(stats['truncation']['gt_128_count'], 3)
        self.assertEqual(stats['truncation']['gt_256_count'], 2)
        self.assertEqual(stats['truncation']['gt_512_count'], 1)
        self.assertEqual(stats['html_br_count'], 1)
        self.assertEqual(stats['duplicate_count'], 1)
        self.assertEqual(stats['empty_count'], 0)

    def test_extract_distinctive_words(self):
        texts = [
            'excellent wonderful brilliant acting',
            'wonderful engaging memorable soundtrack',
            'terrible boring awful disappointment',
            'awful boring waste dreadful',
        ]
        labels = [1, 1, 0, 0]
        res = extract_distinctive_words(texts, labels, top_k=3)
        pos_words = [x['word'] for x in res['top_positive']]
        neg_words = [x['word'] for x in res['top_negative']]
        self.assertIn('wonderful', pos_words)
        self.assertIn('awful', neg_words)
        self.assertTrue(all(x['score'] > 0 for x in res['top_positive']))
        self.assertTrue(all(x['score'] < 0 for x in res['top_negative']))

    def test_error_analysis_and_buckets(self):
        texts = [
            'Great acting!',
            'Boring plot, however great ending',
            'Terrible acting and poor sound',
            'Unusual but memorable film',
        ]
        labels = [1, 0, 0, 1]
        probs = [
            [0.05, 0.95],  # TP
            [0.15, 0.85],  # FP
            [0.92, 0.08],  # TN
            [0.65, 0.35],  # FN
        ]
        token_lengths = [40, 200, 110, 400]
        visible = texts
        tails = ['', 'poor ending', '', 'memorable finale']
        records = build_prediction_records(
            texts, labels, probs, token_lengths, visible, tails, max_length=128
        )
        self.assertEqual([r['outcome'] for r in records], ['TP', 'FP', 'TN', 'FN'])
        self.assertTrue(records[1]['has_contrast_cue'])
        self.assertFalse(records[0]['truncated'])
        self.assertTrue(records[1]['truncated'])

        len_buckets = compute_length_buckets(records)
        self.assertEqual(sum(b['count'] for b in len_buckets), 4)
        conf_buckets = compute_confidence_buckets(records)
        self.assertEqual(sum(b['count'] for b in conf_buckets), 4)

        summary = summarize_error_analysis(records)
        self.assertEqual(summary['confusion_matrix'], {'tn': 1, 'fp': 1, 'fn': 1, 'tp': 1})
        self.assertAlmostEqual(summary['overall']['accuracy'], 0.5)
        self.assertAlmostEqual(summary['truncation_impact']['not_truncated']['accuracy'], 1.0)
        self.assertAlmostEqual(summary['truncation_impact']['truncated']['accuracy'], 0.0)


class ResultsConsistencyTests(unittest.TestCase):
    def test_saved_results_match_reported_metrics(self):
        results_dir = Path(__file__).resolve().parent.parent / 'results'
        metrics = json.loads((results_dir / 'metrics.json').read_text(encoding='utf-8'))
        eda = json.loads((results_dir / 'eda_summary.json').read_text(encoding='utf-8'))
        err = json.loads((results_dir / 'error_analysis.json').read_text(encoding='utf-8'))

        cm = err['confusion_matrix']
        self.assertEqual(cm['tp'] + cm['tn'], 412)
        self.assertEqual(cm['fp'], 64)
        self.assertEqual(cm['fn'], 24)
        self.assertAlmostEqual(metrics['test_accuracy'], 0.824, places=4)
        self.assertAlmostEqual(err['overall']['accuracy'], metrics['test_accuracy'], places=4)
        self.assertEqual(eda['splits']['train']['count'], 2000)
        self.assertEqual(eda['splits']['validation']['count'], 500)
        self.assertEqual(eda['splits']['test']['count'], 500)

        for img_name in [
            'eda_class_and_length.png',
            'eda_text_insights.png',
            'error_analysis_overview.png',
            'correct_vs_wrong_cases.png',
        ]:
            img_path = results_dir / img_name
            self.assertTrue(img_path.is_file(), f'Missing figure: {img_path}')
            self.assertGreater(img_path.stat().st_size, 1024)


if __name__ == '__main__':
    unittest.main()
