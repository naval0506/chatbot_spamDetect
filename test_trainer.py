"""
Tests unitaires pour le ModelTrainer et le ChatbotBanking.
Exécuter : python -m pytest test_trainer.py -v
"""

import sys, os
sys.path.insert(0, "src")
from trainer import ModelTrainer
import pandas as pd


def make_sample_data(n_train=200, n_test=50):
    """Load a small sample from the real dataset for testing."""
    df_train = pd.read_csv("data/train.csv").head(n_train)
    df_test = pd.read_csv("data/test.csv").head(n_test)
    return df_train, df_test


def test_train_returns_metrics():
    """Verify that train() returns all expected metric keys."""
    trainer = ModelTrainer()
    df_train, df_test = make_sample_data()
    metrics = trainer.train(df_train, df_test)

    assert isinstance(metrics, dict)
    for key in ["accuracy", "precision", "recall", "f1_score"]:
        assert key in metrics, f"Missing key: {key}"
        assert 0 <= metrics[key] <= 1, f"{key} out of range: {metrics[key]}"
    print(f"✅ test_train_returns_metrics — {metrics}")


def test_confusion_matrix_exists():
    """Verify that a confusion matrix is computed after training."""
    trainer = ModelTrainer()
    df_train, df_test = make_sample_data()
    trainer.train(df_train, df_test)

    assert trainer.cm is not None, "Confusion matrix is None"
    assert trainer.cm.shape[0] > 0, "Confusion matrix is empty"
    assert len(trainer.classes) > 0, "Classes list is empty"
    print(f"✅ test_confusion_matrix — shape={trainer.cm.shape}, classes={len(trainer.classes)}")


def test_per_class_metrics():
    """Verify that per-class metrics DataFrame is populated."""
    trainer = ModelTrainer()
    df_train, df_test = make_sample_data()
    trainer.train(df_train, df_test)

    assert trainer.per_class_metrics is not None
    assert len(trainer.per_class_metrics) > 0
    for col in ["category", "precision", "recall", "f1_score", "support"]:
        assert col in trainer.per_class_metrics.columns, f"Missing column: {col}"
    print(f"✅ test_per_class_metrics — {len(trainer.per_class_metrics)} categories")


def test_save_and_load(tmp_path="data/test_model_tmp.pkl"):
    """Verify save/load round-trip preserves metrics."""
    trainer = ModelTrainer()
    df_train, df_test = make_sample_data()
    trainer.train(df_train, df_test)
    trainer.save(tmp_path)

    trainer2 = ModelTrainer()
    loaded_metrics = trainer2.load(tmp_path)

    assert loaded_metrics == trainer.metrics
    os.remove(tmp_path)
    print(f"✅ test_save_and_load — metrics match after round-trip")


def test_cross_validation():
    """Verify that cross-validation returns expected keys."""
    trainer = ModelTrainer()
    df_train, _ = make_sample_data(n_train=300)
    X, y = df_train['text'], df_train['label']

    cv_metrics = trainer.validate(X, y, cv=3)

    assert isinstance(cv_metrics, dict)
    for key in ["accuracy_mean", "accuracy_std", "precision_mean",
                 "recall_mean", "f1_mean", "fold_accuracies"]:
        assert key in cv_metrics, f"Missing key: {key}"
    assert len(cv_metrics['fold_accuracies']) == 3
    print(f"✅ test_cross_validation — {cv_metrics}")


def test_export_report():
    """Verify that export_report creates a file."""
    trainer = ModelTrainer()
    df_train, df_test = make_sample_data()
    trainer.train(df_train, df_test)

    report_path = trainer.export_report("data/test_report.md")
    assert os.path.exists(report_path), "Report file not created"
    with open(report_path) as f:
        content = f.read()
    assert "Training Report" in content
    os.remove(report_path)
    print(f"✅ test_export_report — report generated and cleaned up")


def test_chatbot_repondre():
    """Verify that ChatbotBanking.repondre returns expected structure."""
    from chatbot_model import ChatbotBanking
    bot = ChatbotBanking()
    bot.initialiser()

    result = bot.repondre("How do I activate my card?")
    assert isinstance(result, dict)
    for key in ["reponse", "source", "score", "categorie"]:
        assert key in result, f"Missing key: {key}"
    assert len(result["reponse"]) > 0
    print(f"✅ test_chatbot_repondre — source={result['source']}, score={result['score']:.2f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Running all tests...")
    print("=" * 60)
    test_train_returns_metrics()
    test_confusion_matrix_exists()
    test_per_class_metrics()
    test_save_and_load()
    test_cross_validation()
    test_export_report()
    test_chatbot_repondre()
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("=" * 60)
