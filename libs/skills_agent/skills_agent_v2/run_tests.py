#!/usr/bin/env python3
"""
Standalone test runner - works without pytest.
Run with: python3 run_tests.py
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from skills_agent_v2.models import (
    Skill, SkillMetadata, SkillStatus, SkillStep,
    parse_frontmatter, parse_workflow_steps,
)
from skills_agent_v2.loader import SkillLoader
from skills_agent_v2.executor import SkillExecutor, WorkflowResult
from skills_agent_v2.config import Settings, LLMSettings, LogSettings
from skills_agent_v2.exceptions import (
    SkillNotFoundError, SkillLoadError, LLMCallError,
    StepExecutionError, MaxIterationsError,
)
from skills_agent_v2.state import StateManager, StateSnapshot, StateHistory
from skills_agent_v2.log import setup_logging, get_logger

# =====================================================================
# Test Data
# =====================================================================

SAMPLE_SKILL_MD = """\
---
name: test-skill
description: A test skill for unit testing
version: 2.0.0
tags: [test, demo]
inputs: [data]
outputs: [report]
---

# Test Skill

This is a test skill.

## Core Workflow

```
Step 1: Load the input data
Step 2: Process the data
Step 3: Generate the report
```

## Instructions

Follow the steps above to complete the task.
"""

SAMPLE_SKILL_CN = """\
---
name: cn-skill
description: 中文技能测试
version: 1.0.0
tags: [中文, 测试]
---

# 中文技能

## 核心工作流

```
步骤1: 识别场景类型
步骤2: 读取对应场景知识库
步骤3: 大文档分块策略确定
步骤4: 基于知识库枚举进行智能匹配提取
步骤5: 输出统一JSON格式
```

## 说明

按照上述步骤执行。
"""


def setup_skill_dir() -> Path:
    """Create a temporary skill directory with test skills."""
    tmp = Path(tempfile.mkdtemp())

    # Skill 1
    s1 = tmp / "test-skill"
    s1.mkdir()
    (s1 / "SKILL.md").write_text(SAMPLE_SKILL_MD, encoding="utf-8")
    (s1 / "references").mkdir()
    (s1 / "references" / "guide.md").write_text("# Reference Guide\nSome reference content.", encoding="utf-8")
    (s1 / "scripts").mkdir()
    (s1 / "scripts" / "process.py").write_text('print("Hello from script")', encoding="utf-8")

    # Skill 2
    s2 = tmp / "cn-skill"
    s2.mkdir()
    (s2 / "SKILL.md").write_text(SAMPLE_SKILL_CN, encoding="utf-8")

    return tmp


# =====================================================================
# Test Runner
# =====================================================================

passed = 0
failed = 0
errors = []


def test(name):
    """Decorator for test functions."""
    def decorator(fn):
        global passed, failed
        try:
            fn()
            passed += 1
            print(f"  ✅ {name}")
        except AssertionError as e:
            failed += 1
            errors.append((name, str(e)))
            print(f"  ❌ {name}: {e}")
        except Exception as e:
            failed += 1
            errors.append((name, traceback.format_exc()))
            print(f"  ❌ {name}: {e}")
        return fn
    return decorator


# =====================================================================
# Tests
# =====================================================================

print("\n📋 Skills Agent Test Suite\n" + "=" * 50)

print("\n--- Config Tests ---")

@test("Settings: defaults")
def _():
    s = Settings()
    assert s.llm.model == "openai:gpt-4o"
    assert s.llm.temperature == 0.0
    assert s.log.level == "INFO"
    assert s.agent.max_iterations == 25

@test("LLMSettings: provider/model_name extraction")
def _():
    s = LLMSettings(model="anthropic:claude-sonnet-4-5-20250929")
    assert s.provider == "anthropic"
    assert s.model_name == "claude-sonnet-4-5-20250929"
    s2 = LLMSettings(model="gpt-4o")
    assert s2.provider == "openai"
    assert s2.model_name == "gpt-4o"

@test("Settings.from_env")
def _():
    os.environ["SKILLS_AGENT_LLM_MODEL"] = "anthropic:claude-test"
    os.environ["SKILLS_AGENT_LOG_LEVEL"] = "DEBUG"
    os.environ["SKILLS_AGENT_AGENT_MAX_ITERATIONS"] = "50"
    try:
        s = Settings.from_env()
        assert s.llm.model == "anthropic:claude-test"
        assert s.log.level == "DEBUG"
        assert s.agent.max_iterations == 50
    finally:
        del os.environ["SKILLS_AGENT_LLM_MODEL"]
        del os.environ["SKILLS_AGENT_LOG_LEVEL"]
        del os.environ["SKILLS_AGENT_AGENT_MAX_ITERATIONS"]

@test("Settings._from_dict")
def _():
    s = Settings._from_dict({
        "llm": {"model": "test:model", "temperature": 0.5},
        "agent": {"max_iterations": 10},
        "skill_dirs": ["/a", "/b"],
    })
    assert s.llm.model == "test:model"
    assert s.llm.temperature == 0.5
    assert s.agent.max_iterations == 10
    assert s.skill_dirs == ["/a", "/b"]


print("\n--- Exception Tests ---")

@test("SkillNotFoundError: message and fields")
def _():
    e = SkillNotFoundError("foo", ["bar", "baz"])
    assert "foo" in str(e)
    assert e.name == "foo"
    assert e.available == ["bar", "baz"]

@test("LLMCallError: with retries")
def _():
    e = LLMCallError(reason="timeout", retries=3)
    assert "3 retries" in str(e)
    assert "timeout" in str(e)

@test("StepExecutionError")
def _():
    e = StepExecutionError(step_index=2, reason="script crashed")
    assert "Step 2" in str(e)
    assert e.step_index == 2

@test("MaxIterationsError")
def _():
    e = MaxIterationsError(25)
    assert "25" in str(e)


print("\n--- State Tests ---")

@test("StateManager.create: basic")
def _():
    s = StateManager.create("hello")
    assert len(s["messages"]) == 1
    assert s["skill_workflow_status"] == "idle"
    assert s["iteration_count"] == 0
    assert s["metadata"] == {}

@test("StateManager.create: with overrides")
def _():
    s = StateManager.create("hi", active_skill="test", iteration_count=5)
    assert s["active_skill"] == "test"
    assert s["iteration_count"] == 5

@test("StateManager.validate_transition")
def _():
    assert StateManager.validate_transition("idle", "planning") is True
    assert StateManager.validate_transition("idle", "executing") is True
    assert StateManager.validate_transition("idle", "completed") is False
    assert StateManager.validate_transition("executing", "completed") is True
    assert StateManager.validate_transition("executing", "failed") is True

@test("StateManager.to_dict / to_json")
def _():
    s = StateManager.create("test message")
    d = StateManager.to_dict(s)
    assert d["messages"][0]["role"] == "human"
    assert d["messages"][0]["content"] == "test message"
    j = StateManager.to_json(s)
    assert '"human"' in j

@test("StateSnapshot.capture")
def _():
    s = StateManager.create("snapshot test")
    snap = StateSnapshot.capture(s)
    assert snap.iteration == 0
    assert snap.workflow_status == "idle"
    assert snap.message_count == 1

@test("StateHistory: record and summary")
def _():
    history = StateHistory(max_size=5)
    for i in range(3):
        s = StateManager.create(f"msg {i}")
        s["iteration_count"] = i
        history.record(s)
    assert len(history.snapshots) == 3
    summary = history.summary()
    assert "3 snapshots" in summary


print("\n--- Log Tests ---")

@test("setup_logging: no crash")
def _():
    # Just verify it doesn't crash
    import skills_agent.log as log_mod
    log_mod._is_setup = False  # reset
    logger = setup_logging(LogSettings(level="WARNING", format="text"))
    assert logger is not None
    log_mod._is_setup = False  # reset for other tests

@test("get_logger: returns namespaced logger")
def _():
    lg = get_logger("test_module")
    assert "skills_agent" in lg.name


print("\n--- Model Tests ---")

@test("parse_frontmatter: basic")
def _():
    fm, body = parse_frontmatter(SAMPLE_SKILL_MD)
    assert fm["name"] == "test-skill"
    assert fm["description"] == "A test skill for unit testing"
    assert fm["version"] == "2.0.0"
    assert fm["tags"] == ["test", "demo"]
    assert "# Test Skill" in body

@test("parse_frontmatter: no frontmatter")
def _():
    content = "# Just markdown\nNo frontmatter."
    fm, body = parse_frontmatter(content)
    assert fm == {}
    assert body == content

@test("parse_frontmatter: chinese")
def _():
    fm, body = parse_frontmatter(SAMPLE_SKILL_CN)
    assert fm["name"] == "cn-skill"
    assert fm["description"] == "中文技能测试"
    assert "核心工作流" in body

@test("parse_workflow_steps: english")
def _():
    _, body = parse_frontmatter(SAMPLE_SKILL_MD)
    steps = parse_workflow_steps(body)
    assert len(steps) == 3, f"Expected 3 steps, got {len(steps)}"
    assert steps[0].index == 1
    assert "Load" in steps[0].description
    assert steps[2].index == 3

@test("parse_workflow_steps: chinese (步骤)")
def _():
    _, body = parse_frontmatter(SAMPLE_SKILL_CN)
    steps = parse_workflow_steps(body)
    assert len(steps) == 5, f"Expected 5 steps, got {len(steps)}: {[s.description for s in steps]}"
    assert "识别" in steps[0].description
    assert steps[4].index == 5

@test("parse_workflow_steps: no workflow section")
def _():
    steps = parse_workflow_steps("# Some content\nNo workflow here.")
    assert len(steps) == 0

@test("Skill.summary")
def _():
    meta = SkillMetadata(name="test", description="A test", tags=["demo"])
    skill = Skill(metadata=meta)
    s = skill.summary()
    assert "test" in s
    assert "demo" in s

@test("Skill.id")
def _():
    meta = SkillMetadata(name="my-skill")
    skill = Skill(metadata=meta)
    assert skill.id == "my-skill"


print("\n--- Loader Tests ---")
skill_dir = setup_skill_dir()

@test("SkillLoader.discover")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    discovered = loader.discover()
    assert len(discovered) == 2, f"Expected 2, got {len(discovered)}"
    names = {s.metadata.name for s in discovered}
    assert "test-skill" in names
    assert "cn-skill" in names

@test("SkillLoader.load_skill")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    skill = loader.load_skill("test-skill")
    assert skill.status == SkillStatus.LOADED
    assert len(skill.workflow_steps) == 3
    assert "guide.md" in skill.reference_files
    assert "process.py" in skill.scripts

@test("SkillLoader.load_skill: nonexistent raises KeyError")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    try:
        loader.load_skill("nonexistent")
        assert False, "Should have raised KeyError"
    except KeyError:
        pass

@test("SkillLoader.match_skill")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    match = loader.match_skill("I need to test something")
    assert match is not None
    assert match.metadata.name == "test-skill"

@test("SkillLoader.load_from_content")
def _():
    loader = SkillLoader()
    skill = loader.load_from_content("inline-test", SAMPLE_SKILL_MD)
    assert skill.id == "test-skill"
    assert skill.status == SkillStatus.LOADED
    assert len(skill.workflow_steps) == 3

@test("SkillLoader.get_skills_context: progressive disclosure")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    ctx = loader.get_skills_context()
    assert "test-skill" in ctx
    assert "<available_skills>" in ctx
    assert "Core Workflow" not in ctx  # Body not included in summary


print("\n--- Executor Tests ---")

@test("SkillExecutor: empty workflow")
def _():
    executor = SkillExecutor()
    meta = SkillMetadata(name="empty")
    skill = Skill(metadata=meta, status=SkillStatus.LOADED)
    result = executor.execute_workflow(skill)
    assert result.success

@test("SkillExecutor: with LLM callback")
def _():
    def fake_llm(system, user):
        return f"Processed: {user}"
    executor = SkillExecutor(llm_callback=fake_llm)
    meta = SkillMetadata(name="test", description="test")
    skill = Skill(
        metadata=meta, body="Test body", status=SkillStatus.LOADED,
        workflow_steps=[
            SkillStep(index=1, description="Do step 1"),
            SkillStep(index=2, description="Do step 2"),
        ],
    )
    result = executor.execute_workflow(skill)
    assert result.success
    assert result.completed_steps == 2
    assert "Processed" in result.step_results[0].output

@test("SkillExecutor: read_reference action")
def _():
    executor = SkillExecutor()
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    skill = loader.load_skill("test-skill")
    step = SkillStep(index=1, description="Read guide", action="read_reference",
                     params={"reference": "guide.md"})
    result = executor._execute_step(step, skill, {}, {})
    assert result.success
    assert "Reference Guide" in result.output

@test("SkillExecutor: run_script action")
def _():
    executor = SkillExecutor(work_dir=str(skill_dir / "workspace"))
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    skill = loader.load_skill("test-skill")
    step = SkillStep(index=1, description="Run script", action="run_script",
                     params={"script": "process.py"})
    result = executor._execute_step(step, skill, {}, {})
    assert result.success, f"Script failed: {result.error}"
    assert "Hello from script" in result.output

@test("WorkflowResult.summary")
def _():
    r1 = WorkflowResult(skill_name="test", success=True)
    assert "✅" in r1.summary()
    r2 = WorkflowResult(skill_name="test", success=False)
    assert "❌" in r2.summary()


print("\n--- New Loader Features ---")

@test("SkillLoader: constructor with skill_paths")
def _():
    s1 = skill_dir / "test-skill"
    s2 = skill_dir / "cn-skill"
    loader = SkillLoader(skill_paths=[str(s1), str(s2)])
    assert len(loader) == 2
    assert "test-skill" in loader
    assert "cn-skill" in loader

@test("SkillLoader.add_skill_path: by directory")
def _():
    loader = SkillLoader()
    skill = loader.add_skill_path(str(skill_dir / "test-skill"))
    assert skill is not None
    assert skill.id == "test-skill"
    assert skill.status == SkillStatus.DISCOVERED

@test("SkillLoader.add_skill_path: by SKILL.md file")
def _():
    loader = SkillLoader()
    skill = loader.add_skill_path(str(skill_dir / "test-skill" / "SKILL.md"))
    assert skill is not None
    assert skill.id == "test-skill"

@test("SkillLoader.add_skill_path: auto_load=True")
def _():
    loader = SkillLoader()
    skill = loader.add_skill_path(str(skill_dir / "test-skill"), auto_load=True)
    assert skill is not None
    assert skill.status == SkillStatus.LOADED
    assert len(skill.workflow_steps) == 3
    assert "guide.md" in skill.reference_files

@test("SkillLoader.add_skill_path: nonexistent path returns None")
def _():
    loader = SkillLoader()
    skill = loader.add_skill_path("/nonexistent/path/to/skill")
    assert skill is None

@test("SkillLoader.add_skill_path: duplicate skill skips")
def _():
    loader = SkillLoader()
    s1 = loader.add_skill_path(str(skill_dir / "test-skill"))
    s2 = loader.add_skill_path(str(skill_dir / "test-skill"))
    assert s1 is not None
    assert s2 is not None  # returns existing
    assert len(loader) == 1  # not duplicated

@test("SkillLoader.load_skill: by filesystem path (auto-register)")
def _():
    loader = SkillLoader()
    # load_skill with a path that isn't registered yet
    skill = loader.load_skill(str(skill_dir / "cn-skill"))
    assert skill.id == "cn-skill"
    assert skill.status == SkillStatus.LOADED
    assert len(skill.workflow_steps) == 5

@test("SkillLoader.load_skill: by path (SKILL.md file)")
def _():
    loader = SkillLoader()
    skill = loader.load_skill(str(skill_dir / "test-skill" / "SKILL.md"))
    assert skill.id == "test-skill"
    assert skill.status == SkillStatus.LOADED

@test("SkillLoader: multi-dir + single-path mixed")
def _():
    # Create an extra standalone skill
    extra = skill_dir / "extra-skill"
    extra.mkdir(exist_ok=True)
    (extra / "SKILL.md").write_text(
        "---\nname: extra\ndescription: Extra skill\n---\n# Extra\n## Workflow\n",
        encoding="utf-8",
    )
    loader = SkillLoader(
        skill_dirs=[str(skill_dir / "test-skill")],   # only test-skill dir
        skill_paths=[str(extra)],                       # single extra skill
    )
    loader.discover()  # discovers test-skill from dir
    # extra was already added by skill_paths
    assert "test-skill" in loader
    assert "extra" in loader
    assert len(loader) == 2

@test("SkillLoader.reload_skill")
def _():
    loader = SkillLoader()
    loader.add_skill_path(str(skill_dir / "test-skill"), auto_load=True)
    skill = loader.get_skill("test-skill")
    assert skill.status == SkillStatus.LOADED
    # Reload
    reloaded = loader.reload_skill("test-skill")
    assert reloaded.status == SkillStatus.LOADED
    assert len(reloaded.workflow_steps) == 3

@test("SkillLoader.remove_skill")
def _():
    loader = SkillLoader(skill_paths=[str(skill_dir / "test-skill")])
    assert "test-skill" in loader
    assert loader.remove_skill("test-skill") is True
    assert "test-skill" not in loader
    assert loader.remove_skill("nonexistent") is False

@test("SkillLoader.has_skill / list_skill_names / list_dirs")
def _():
    loader = SkillLoader(
        skill_dirs=[str(skill_dir)],
        skill_paths=[str(skill_dir / "test-skill")],
    )
    loader.discover()
    assert loader.has_skill("test-skill")
    assert "test-skill" in loader.list_skill_names()
    assert len(loader.list_dirs()) == 1

@test("SkillLoader.match_skills (top_k)")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    results = loader.match_skills("test demo")
    assert len(results) >= 1
    assert results[0][0].metadata.name == "test-skill"
    assert results[0][1] > 0  # has a score

@test("SkillLoader.__getitem__ / __len__ / __repr__")
def _():
    loader = SkillLoader()
    loader.add_skill_path(str(skill_dir / "test-skill"))
    loader.add_skill_path(str(skill_dir / "cn-skill"))
    assert len(loader) == 2
    assert loader["test-skill"].id == "test-skill"
    assert "SkillLoader" in repr(loader)
    try:
        _ = loader["nonexistent"]
        assert False, "Should raise KeyError"
    except KeyError:
        pass

@test("SkillLoader.discover_and_load_all")
def _():
    fresh_dir = setup_skill_dir()  # fresh temp dir
    loader = SkillLoader(skill_dirs=[str(fresh_dir)])
    loaded = loader.discover_and_load_all()
    assert len(loaded) == 2
    for s in loader.list_skills():
        assert s.status == SkillStatus.LOADED

@test("SkillLoader.clear")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    assert len(loader) > 0
    loader.clear()
    assert len(loader) == 0
    assert len(loader.list_dirs()) == 0


print("\n--- Integration Tests ---")

@test("Full flow: discover → load → execute (no LLM)")
def _():
    fresh_dir = setup_skill_dir()
    loader = SkillLoader(skill_dirs=[str(fresh_dir)])
    loader.discover()
    assert len(loader.list_skills()) == 2

    skill = loader.load_skill("test-skill")
    assert skill.status == SkillStatus.LOADED
    assert len(skill.workflow_steps) == 3

    executor = SkillExecutor()
    result = executor.execute_workflow(skill)
    assert result.success
    assert result.completed_steps == 3

@test("Chinese skill: discover → load → verify steps")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
    loader.discover()
    skill = loader.load_skill("cn-skill")
    assert len(skill.workflow_steps) == 5
    assert "识别" in skill.workflow_steps[0].description


# =====================================================================
# Summary
# =====================================================================

print(f"\n{'=' * 50}")
print(f"Results: {passed} passed, {failed} failed")
if errors:
    print(f"\nFailed tests:")
    for name, err in errors:
        print(f"  • {name}")
        for line in err.split("\n")[:3]:
            print(f"    {line}")
print()

sys.exit(0 if failed == 0 else 1)
