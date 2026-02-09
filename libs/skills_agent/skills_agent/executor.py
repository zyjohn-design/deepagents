"""
Skill Executor - Runs skill workflows step by step.

Handles:
  - Sequential and dependency-based step execution
  - LLM-driven step interpretation
  - Script execution (Python, bash)
  - Reference file reading on demand
  - Progress tracking
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .models import Skill, SkillStatus, SkillStep

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a single skill step."""
    step_index: int
    success: bool
    output: str = ""
    error: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of executing an entire skill workflow."""
    skill_name: str
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    final_output: str = ""
    error: str = ""

    @property
    def completed_steps(self) -> int:
        return sum(1 for r in self.step_results if r.success)

    @property
    def total_steps(self) -> int:
        return len(self.step_results)

    def summary(self) -> str:
        status = "✅ Success" if self.success else "❌ Failed"
        return (
            f"Workflow [{self.skill_name}] {status} "
            f"({self.completed_steps}/{self.total_steps} steps completed)"
        )


class SkillExecutor:
    """Executes skill workflows with support for various action types."""

    def __init__(
        self,
        llm_callback: Callable[[str, str], str] | None = None,
        work_dir: str | Path | None = None,
    ):
        """
        Args:
            llm_callback: A function(system_prompt, user_message) -> str
                          for steps that need LLM reasoning.
            work_dir: Working directory for script execution.
        """
        self._llm_callback = llm_callback
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self._work_dir.mkdir(parents=True, exist_ok=True)

    def execute_workflow(
        self,
        skill: Skill,
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute all workflow steps in a skill sequentially."""
        if not skill.workflow_steps:
            return WorkflowResult(
                skill_name=skill.id,
                success=True,
                final_output="No workflow steps defined. Skill loaded for reference only.",
            )

        skill.status = SkillStatus.EXECUTING
        result = WorkflowResult(skill_name=skill.id, success=True)
        ctx = context or {}
        step_outputs: dict[int, StepResult] = {}

        for step in sorted(skill.workflow_steps, key=lambda s: s.index):
            # Check dependencies
            for dep in step.depends_on:
                dep_result = step_outputs.get(dep)
                if not dep_result or not dep_result.success:
                    step_result = StepResult(
                        step_index=step.index,
                        success=False,
                        error=f"Dependency step {dep} not completed",
                    )
                    result.step_results.append(step_result)
                    result.success = False
                    result.error = f"Step {step.index} blocked by dependency {dep}"
                    skill.status = SkillStatus.FAILED
                    return result

            # Execute step
            logger.info("Executing step %d: %s", step.index, step.description)
            step_result = self._execute_step(step, skill, ctx, step_outputs)
            step_outputs[step.index] = step_result
            result.step_results.append(step_result)

            if not step_result.success:
                result.success = False
                result.error = f"Step {step.index} failed: {step_result.error}"
                skill.status = SkillStatus.FAILED
                return result

            # Update context with step artifacts
            ctx.update(step_result.artifacts)

        # Collect final output
        if result.step_results:
            result.final_output = result.step_results[-1].output

        skill.status = SkillStatus.COMPLETED
        return result

    def _execute_step(
        self,
        step: SkillStep,
        skill: Skill,
        context: dict[str, Any],
        prev_results: dict[int, StepResult],
    ) -> StepResult:
        """Execute a single workflow step."""
        action = step.action.lower() if step.action else "llm_reason"

        try:
            if action == "read_file":
                return self._action_read_file(step, skill)
            elif action == "run_script":
                return self._action_run_script(step, skill, context)
            elif action == "read_reference":
                return self._action_read_reference(step, skill)
            elif action in ("llm_call", "llm_reason", ""):
                return self._action_llm_reason(step, skill, context, prev_results)
            else:
                return self._action_llm_reason(step, skill, context, prev_results)
        except Exception as e:
            logger.exception("Step %d execution error", step.index)
            return StepResult(
                step_index=step.index,
                success=False,
                error=str(e),
            )

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _action_read_file(self, step: SkillStep, skill: Skill) -> StepResult:
        """Read a file from the skill directory or filesystem."""
        file_path = step.params.get("path", "")
        if not file_path and skill.path:
            # Default: read from skill directory
            file_path = str(skill.path.parent / step.params.get("file", ""))

        path = Path(file_path)
        if not path.exists():
            return StepResult(
                step_index=step.index,
                success=False,
                error=f"File not found: {file_path}",
            )

        content = path.read_text(encoding="utf-8")
        return StepResult(
            step_index=step.index,
            success=True,
            output=content,
            artifacts={"file_content": content, "file_path": str(path)},
        )

    def _action_read_reference(self, step: SkillStep, skill: Skill) -> StepResult:
        """Read a reference file from the skill's references."""
        ref_name = step.params.get("reference", "")
        if ref_name in skill.reference_files:
            content = skill.reference_files[ref_name]
            return StepResult(
                step_index=step.index,
                success=True,
                output=content,
                artifacts={"reference_content": content, "reference_name": ref_name},
            )
        return StepResult(
            step_index=step.index,
            success=False,
            error=f"Reference '{ref_name}' not found. Available: {list(skill.reference_files.keys())}",
        )

    def _action_run_script(
        self, step: SkillStep, skill: Skill, context: dict[str, Any]
    ) -> StepResult:
        """Execute a Python or shell script from the skill."""
        script_name = step.params.get("script", "")

        if script_name not in skill.scripts:
            return StepResult(
                step_index=step.index,
                success=False,
                error=f"Script '{script_name}' not found. Available: {list(skill.scripts.keys())}",
            )

        script_content = skill.scripts[script_name]

        if script_name.endswith(".py"):
            return self._run_python_script(step.index, script_content, context)
        elif script_name.endswith(".sh"):
            return self._run_shell_script(step.index, script_content, context)
        else:
            return StepResult(
                step_index=step.index,
                success=False,
                error=f"Unsupported script type: {script_name}",
            )

    def _run_python_script(
        self, step_index: int, script: str, context: dict[str, Any]
    ) -> StepResult:
        """Execute Python script in a subprocess."""
        script_path = self._work_dir / f"step_{step_index}.py"
        script_path.write_text(script, encoding="utf-8")

        try:
            result = subprocess.run(
                ["python3", str(script_path)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self._work_dir),
            )
            return StepResult(
                step_index=step_index,
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
            )
        except subprocess.TimeoutExpired:
            return StepResult(
                step_index=step_index,
                success=False,
                error="Script execution timed out (120s)",
            )

    def _run_shell_script(
        self, step_index: int, script: str, context: dict[str, Any]
    ) -> StepResult:
        """Execute shell script in a subprocess."""
        try:
            result = subprocess.run(
                ["bash", "-c", script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(self._work_dir),
            )
            return StepResult(
                step_index=step_index,
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr if result.returncode != 0 else "",
            )
        except subprocess.TimeoutExpired:
            return StepResult(
                step_index=step_index,
                success=False,
                error="Script execution timed out (60s)",
            )

    def _action_llm_reason(
        self,
        step: SkillStep,
        skill: Skill,
        context: dict[str, Any],
        prev_results: dict[int, StepResult],
    ) -> StepResult:
        """Use LLM to reason about a workflow step."""
        if not self._llm_callback:
            return StepResult(
                step_index=step.index,
                success=True,
                output=f"[LLM reasoning needed] Step {step.index}: {step.description}",
            )

        # Build context from previous steps
        prev_context = ""
        for idx, res in sorted(prev_results.items()):
            if res.success and res.output:
                prev_context += f"\n--- Step {idx} output ---\n{res.output}\n"

        system_prompt = (
            f"You are executing a skill workflow: {skill.metadata.name}\n"
            f"Skill description: {skill.metadata.description}\n\n"
            f"Full skill instructions:\n{skill.body}\n\n"
            f"Previous step results:{prev_context}\n\n"
            f"Additional context: {context}\n"
        )

        user_message = (
            f"Execute step {step.index}: {step.description}\n"
            f"Provide the output for this step."
        )

        try:
            output = self._llm_callback(system_prompt, user_message)
            return StepResult(
                step_index=step.index,
                success=True,
                output=output,
            )
        except Exception as e:
            return StepResult(
                step_index=step.index,
                success=False,
                error=f"LLM call failed: {e}",
            )
