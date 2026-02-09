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

from skills_agent.models import (
    Skill, SkillMetadata, SkillStatus, SkillStep,
    parse_frontmatter, parse_workflow_steps,
)
from skills_agent.loader import SkillLoader
from skills_agent.executor import SkillExecutor, WorkflowResult

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


print("\n--- Integration Tests ---")

@test("Full flow: discover → load → execute (no LLM)")
def _():
    loader = SkillLoader(skill_dirs=[str(skill_dir)])
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
