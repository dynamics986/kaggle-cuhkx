from pathlib import Path

from cuhkx_sep12.serial import Runner, exclusive_lock


def test_failure_continues_and_success_is_reused(tmp_path):
    runner = Runner(tmp_path / "run")
    assert runner.job("bad", "module_that_does_not_exist_sep12", lambda out: []) is None
    result = runner.job(
        "good", "venv", lambda out: ["--without-pip", out], required=("pyvenv.cfg",)
    )
    assert result and (result / "pyvenv.cfg").is_file()
    assert runner.state["jobs"]["bad"]["status"] == "failed"
    assert runner.state["jobs"]["good"]["status"] == "success"
    reused = runner.job("good", "must_not_execute", lambda out: [], required=("pyvenv.cfg",))
    assert reused == result
    runner.job("bad", "module_that_does_not_exist_sep12", lambda out: [])
    assert runner.state["jobs"]["bad"]["attempt"] == 2
    assert Path(runner.state["jobs"]["bad"]["log"]).is_file()


def test_blocked_dependency_does_not_run(tmp_path):
    runner = Runner(tmp_path / "run")
    assert runner.job("blocked", "venv", lambda out: [], dependencies=False) is None
    assert runner.state["jobs"]["blocked"]["status"] == "blocked"


def test_lock_is_released(tmp_path):
    with exclusive_lock(tmp_path / "lock"):
        pass
    with exclusive_lock(tmp_path / "lock"):
        pass


def test_all_eight_methods_attempted_after_failures(tmp_path, monkeypatch):
    import json

    from cuhkx_sep12 import serial

    calls = []

    def fail(self, name, *args, **kwargs):
        calls.append(name)
        return None

    monkeypatch.setattr(serial.Runner, "job", fail)
    assert serial.run(tmp_path) == 1
    state = json.loads((tmp_path / "status.json").read_text())
    assert state["status"] == "finished"
    assert len(state["methods"]) == 8
    assert "m01_lightgbm" in calls
    assert "m08_se_attention_fold_4" in calls


def test_timeout_marks_failure(tmp_path):
    runner = Runner(tmp_path / 'run', timeout_hours=0.0001)
    result = runner.job('slow', 'timeit',
                        lambda out: ['-n', '1', '-r', '1', 'import time; time.sleep(30)'])
    assert result is None
    assert runner.state['jobs']['slow']['status'] == 'failed'
    assert 'timeout' in runner.state['jobs']['slow']['error']
