from pathlib import Path

from cuhkx_har.monitor import progress_bar, render_dashboard, sparkline


def test_monitor_render_contains_live_and_best_metrics() -> None:
    progress = {
        "phase": "train",
        "epoch": 2,
        "total_epochs": 40,
        "step": 5,
        "total_steps": 10,
        "loss": 1.25,
        "accuracy": 0.5,
        "elapsed_seconds": 60,
    }
    history = [
        {
            "epoch": 1,
            "train_loss": 2.0,
            "valid_loss": 2.2,
            "train_accuracy": 0.3,
            "valid_accuracy": 0.4,
            "learning_rate": 4e-4,
        }
    ]
    output = render_dashboard(Path("run"), progress, history, {}, patience=8)
    assert "Epoch 2/40" in output
    assert "Running loss 1.2500" in output
    assert "Best validation accuracy 0.4000" in output
    assert "early-stop wait 0/8" in output


def test_monitor_chart_helpers() -> None:
    assert progress_bar(5, 10).endswith("50.0%")
    assert sparkline([]) == "waiting"
    assert len(sparkline([3.0, 2.0, 1.0])) == 3
