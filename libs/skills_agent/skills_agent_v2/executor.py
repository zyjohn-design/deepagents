"""
Skill Executor — runs skill workflows step by step.

Built-in actions:
  read_file         Read a file from disk
  read_reference    Read a full reference file
  search_reference  Search within references (keyword/section, saves tokens)
  run_script        Execute Python/shell script (supports dynamic args via templates)
  invoke_skill      Call another skill's workflow (skill composition)
  llm_reason        LLM interprets and executes the step (default)

Template variables in step params:
  ${step.N.output}            → output text of step N
  ${step.N.artifact.KEY}      → specific artifact from step N
  ${context.KEY}              → value from the context dict

Example SKILL.md workflow:
  ## 核心工作流
  步骤1: 查询API认证文档 [action=search_reference, query=认证 token, file=api_doc.md]
  步骤2: 提取参数配置 [action=llm_reason]
  步骤3: 运行转换脚本 [action=run_script, script=transform.py, args=--config ${step.2.output}]
  步骤4: 验证数据 [action=invoke_skill, skill=data-validator, input=${step.3.output}]
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .models import Skill, SkillStatus, SkillStep
from .reference import ReferenceManager

logger = logging.getLogger(__name__)


# ======================================================================
# Result types
# ======================================================================

@dataclass
class StepResult:
    """Result of executing a single skill step."""
    step_index: int
    success: bool
    output: str = ""
    error: str = ""
    action: str = ""
    duration_ms: float = 0.0
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """Result of executing an entire skill workflow."""
    skill_name: str
    success: bool
    step_results: list[StepResult] = field(default_factory=list)
    final_output: str = ""
    error: str = ""
    total_duration_ms: float = 0.0

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
            f"({self.completed_steps}/{self.total_steps} steps, "
            f"{self.total_duration_ms:.0f}ms)"
        )


# Action handler type
ActionHandler = Callable[
    ["SkillStep", "Skill", dict[str, Any], dict[int, "StepResult"]],
    "StepResult",
]


# ======================================================================
# Executor
# ======================================================================

class SkillExecutor:
    """Executes skill workflows (Mode B: pipeline execution).

    Each step's params can use ${...} template variables to reference
    previous step outputs, enabling dynamic parameter passing.
    """

    def __init__(
        self,
        llm_callback: Callable[[str, str], str] | None = None,
        skill_loader: Any | None = None,
        work_dir: str | Path | None = None,
        script_timeout_python: int = 120,
        script_timeout_shell: int = 60,
    ):
        """
        Args:
            llm_callback: (system_prompt, user_message) -> str for LLM steps.
            skill_loader: SkillLoader instance (needed for invoke_skill action).
            work_dir: Working directory for script execution.
            script_timeout_python: Python script timeout (seconds).
            script_timeout_shell: Shell script timeout (seconds).
        """
        self._llm_callback = llm_callback
        self._skill_loader = skill_loader
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp())
        self._work_dir.mkdir(parents=True, exist_ok=True)
        self._timeout_python = script_timeout_python
        self._timeout_shell = script_timeout_shell
        self._actions: dict[str, ActionHandler] = {}
        self._ref_managers: dict[str, ReferenceManager] = {}

        logger.info(
            "SkillExecutor initialized: work_dir=%s, llm=%s, loader=%s",
            self._work_dir, "yes" if llm_callback else "no",
            "yes" if skill_loader else "no",
        )

    def register_action(self, name: str, handler: ActionHandler) -> None:
        """Register a custom action handler."""
        self._actions[name] = handler
        logger.info("Registered custom action: '%s'", name)

    def _get_ref_manager(self, skill: Skill) -> ReferenceManager:
        """Get or create a ReferenceManager for a skill (lazy, cached)."""
        if skill.id not in self._ref_managers:
            mgr = ReferenceManager()
            mgr.index_skill_references(skill)
            self._ref_managers[skill.id] = mgr
        return self._ref_managers[skill.id]

    # ------------------------------------------------------------------
    # Workflow execution
    # ------------------------------------------------------------------

    def execute_workflow(
        self,
        skill: Skill,
        context: dict[str, Any] | None = None,
    ) -> WorkflowResult:
        """Execute all workflow steps sequentially with template resolution."""
        workflow_start = time.monotonic()

        if not skill.workflow_steps:
            logger.info("[%s] No workflow steps, reference-only skill", skill.id)
            return WorkflowResult(
                skill_name=skill.id, success=True,
                final_output="No workflow steps defined.",
            )

        logger.info("━━━ Workflow START: '%s' (%d steps) ━━━", skill.id, len(skill.workflow_steps))
        skill.status = SkillStatus.EXECUTING
        result = WorkflowResult(skill_name=skill.id, success=True)
        ctx = dict(context or {})
        step_outputs: dict[int, StepResult] = {}

        for step in sorted(skill.workflow_steps, key=lambda s: s.index):
            # Dependency check
            for dep in step.depends_on:
                dep_result = step_outputs.get(dep)
                if not dep_result or not dep_result.success:
                    logger.error("[%s] Step %d BLOCKED by dep %d", skill.id, step.index, dep)
                    sr = StepResult(step_index=step.index, success=False,
                                    error=f"Dependency step {dep} not completed", action="dep_check")
                    result.step_results.append(sr)
                    result.success = False
                    result.error = f"Step {step.index} blocked by dep {dep}"
                    skill.status = SkillStatus.FAILED
                    result.total_duration_ms = (time.monotonic() - workflow_start) * 1000
                    return result

            # Resolve template variables in params
            resolved_params = step.resolve_params(ctx, step_outputs)
            logger.info("[%s] Step %d/%d: %s (action=%s)",
                        skill.id, step.index, len(skill.workflow_steps),
                        step.description[:60], step.action or "llm_reason")
            if resolved_params != step.params:
                logger.debug("[%s] Step %d: params resolved: %s → %s",
                             skill.id, step.index, step.params, resolved_params)

            # Execute with resolved params
            resolved_step = SkillStep(
                index=step.index, description=step.description,
                action=step.action, params=resolved_params,
                depends_on=step.depends_on, raw_description=step.raw_description,
            )
            sr = self._execute_step(resolved_step, skill, ctx, step_outputs)
            step_outputs[step.index] = sr
            result.step_results.append(sr)

            if sr.success:
                logger.info("[%s] Step %d ✅ (%s, %dms, %d chars)",
                            skill.id, step.index, sr.action, sr.duration_ms, len(sr.output))
            else:
                logger.error("[%s] Step %d ❌ (%s): %s", skill.id, step.index, sr.action, sr.error)
                result.success = False
                result.error = f"Step {step.index} failed: {sr.error}"
                skill.status = SkillStatus.FAILED
                result.total_duration_ms = (time.monotonic() - workflow_start) * 1000
                return result

            ctx.update(sr.artifacts)

        if result.step_results:
            result.final_output = result.step_results[-1].output
        skill.status = SkillStatus.COMPLETED
        result.total_duration_ms = (time.monotonic() - workflow_start) * 1000
        logger.info("━━━ Workflow DONE: '%s' ✅ (%d/%d, %.0fms) ━━━",
                     skill.id, result.completed_steps, result.total_steps, result.total_duration_ms)
        return result

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------

    def _execute_step(self, step: SkillStep, skill: Skill,
                      context: dict[str, Any], prev: dict[int, StepResult]) -> StepResult:
        action = step.action.lower().strip() if step.action else ""
        start = time.monotonic()
        try:
            if action == "read_file":
                r = self._action_read_file(step, skill)
            elif action == "read_reference":
                r = self._action_read_reference(step, skill)
            elif action == "search_reference":
                r = self._action_search_reference(step, skill)
            elif action == "run_script":
                r = self._action_run_script(step, skill, context)
            elif action == "invoke_skill":
                r = self._action_invoke_skill(step, skill, context, prev)
            elif action in ("llm_call", "llm_reason"):
                r = self._action_llm_reason(step, skill, context, prev)
            elif action in self._actions:
                r = self._actions[action](step, skill, context, prev)
            elif action == "":
                r = self._action_llm_reason(step, skill, context, prev)
            else:
                logger.warning("[%s] Unknown action '%s', fallback to llm_reason", skill.id, action)
                r = self._action_llm_reason(step, skill, context, prev)
            r.action = action or "llm_reason"
            r.duration_ms = (time.monotonic() - start) * 1000
            return r
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            logger.exception("[%s] Step %d EXCEPTION: %s", skill.id, step.index, e)
            return StepResult(step_index=step.index, success=False, error=str(e),
                              action=action or "llm_reason", duration_ms=elapsed)

    # ------------------------------------------------------------------
    # Action: read_file
    # ------------------------------------------------------------------

    def _action_read_file(self, step: SkillStep, skill: Skill) -> StepResult:
        file_path = step.params.get("path", "")
        if not file_path and skill.path:
            file_path = str(skill.path.parent / step.params.get("file", ""))
        path = Path(file_path)
        if not path.exists():
            return StepResult(step_index=step.index, success=False,
                              error=f"File not found: {file_path}")
        content = path.read_text(encoding="utf-8")
        return StepResult(step_index=step.index, success=True, output=content,
                          artifacts={"file_content": content, "file_path": str(path)})

    # ------------------------------------------------------------------
    # Action: read_reference (full file)
    # ------------------------------------------------------------------

    def _action_read_reference(self, step: SkillStep, skill: Skill) -> StepResult:
        ref_name = step.params.get("reference", "")
        if ref_name in skill.reference_files:
            content = skill.reference_files[ref_name]
            return StepResult(step_index=step.index, success=True, output=content,
                              artifacts={"reference_content": content, "reference_name": ref_name})
        return StepResult(step_index=step.index, success=False,
                          error=f"Reference '{ref_name}' not found. Available: {list(skill.reference_files.keys())}")

    # ------------------------------------------------------------------
    # Action: search_reference (keyword search, token-efficient)
    # ------------------------------------------------------------------

    def _action_search_reference(self, step: SkillStep, skill: Skill) -> StepResult:
        """Search within indexed references for specific content.

        Params:
            query:   Search query (keywords or natural language).
            file:    (optional) Restrict to a specific reference file.
            section: (optional) Get exact section by heading.
            top_k:   (optional) Number of results, default 3.
        """
        ref_mgr = self._get_ref_manager(skill)
        query = step.params.get("query", "")
        file_name = step.params.get("file", None)
        section = step.params.get("section", "")
        top_k = int(step.params.get("top_k", "3"))

        # Mode 1: Exact section lookup
        if section and file_name:
            content = ref_mgr.get_section(file_name, section)
            if content:
                logger.debug("[%s] search_reference: exact section '%s' found (%d chars)",
                             skill.id, section, len(content))
                return StepResult(step_index=step.index, success=True, output=content,
                                  artifacts={"matched_section": section})
            return StepResult(step_index=step.index, success=False,
                              error=f"Section '{section}' not found in '{file_name}'. TOC:\n{ref_mgr.get_toc(file_name)}")

        # Mode 2: Keyword search
        if query:
            result_text = ref_mgr.search_text(query, top_k=top_k, file_name=file_name)
            results = ref_mgr.search(query, top_k=top_k, file_name=file_name)
            matched = [r.section.heading for r in results]
            logger.debug("[%s] search_reference: query='%s' → %d hits: %s",
                         skill.id, query, len(results), matched)
            return StepResult(step_index=step.index, success=True, output=result_text,
                              artifacts={"matched_sections": matched, "query": query})

        # Mode 3: Return TOC for navigation
        toc = ref_mgr.get_toc(file_name)
        return StepResult(step_index=step.index, success=True, output=toc,
                          artifacts={"toc": toc})

    # ------------------------------------------------------------------
    # Action: run_script (with dynamic args)
    # ------------------------------------------------------------------

    def _action_run_script(self, step: SkillStep, skill: Skill, context: dict[str, Any]) -> StepResult:
        """Execute script with optional dynamic arguments.

        Params:
            script: Script filename (e.g. "transform.py")
            args:   (optional) Command-line arguments string
            env:    (optional) Extra environment variables as "K=V,K2=V2"
        """
        script_name = step.params.get("script", "")
        extra_args = step.params.get("args", "")
        extra_env_str = step.params.get("env", "")

        if script_name not in skill.scripts:
            return StepResult(step_index=step.index, success=False,
                              error=f"Script '{script_name}' not found. Available: {list(skill.scripts.keys())}")

        script_content = skill.scripts[script_name]

        # Parse extra environment variables
        env = None
        if extra_env_str:
            import os
            env = dict(os.environ)
            for pair in extra_env_str.split(","):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    env[k.strip()] = v.strip()

        if script_name.endswith(".py"):
            return self._run_python_script(step.index, script_name, script_content, extra_args, env)
        elif script_name.endswith(".sh"):
            return self._run_shell_script(step.index, script_name, script_content, extra_args, env)
        return StepResult(step_index=step.index, success=False,
                          error=f"Unsupported script type: {script_name}")

    def _run_python_script(self, step_index: int, name: str, script: str,
                           args: str, env: dict | None) -> StepResult:
        script_path = self._work_dir / f"step_{step_index}_{name}"
        script_path.write_text(script, encoding="utf-8")
        cmd = ["python3", str(script_path)]
        if args:
            cmd.extend(args.split())
        logger.debug("Running: %s", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=self._timeout_python, cwd=str(self._work_dir), env=env)
            return StepResult(step_index=step_index, success=proc.returncode == 0,
                              output=proc.stdout, error=proc.stderr if proc.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return StepResult(step_index=step_index, success=False,
                              error=f"Script '{name}' timed out ({self._timeout_python}s)")

    def _run_shell_script(self, step_index: int, name: str, script: str,
                          args: str, env: dict | None) -> StepResult:
        full_script = f"{script} {args}" if args else script
        try:
            proc = subprocess.run(["bash", "-c", full_script], capture_output=True, text=True,
                                  timeout=self._timeout_shell, cwd=str(self._work_dir), env=env)
            return StepResult(step_index=step_index, success=proc.returncode == 0,
                              output=proc.stdout, error=proc.stderr if proc.returncode != 0 else "")
        except subprocess.TimeoutExpired:
            return StepResult(step_index=step_index, success=False,
                              error=f"Script '{name}' timed out ({self._timeout_shell}s)")

    # ------------------------------------------------------------------
    # Action: invoke_skill (skill composition)
    # ------------------------------------------------------------------

    def _action_invoke_skill(self, step: SkillStep, skill: Skill,
                             context: dict[str, Any], prev: dict[int, StepResult]) -> StepResult:
        """Invoke another skill's workflow.

        Params:
            skill:  Name of the sub-skill to invoke.
            input:  (optional) Input data to pass as context.
        """
        sub_skill_name = step.params.get("skill", "")
        input_data = step.params.get("input", "")

        if not self._skill_loader:
            return StepResult(step_index=step.index, success=False,
                              error="invoke_skill requires a SkillLoader. Pass skill_loader to SkillExecutor.")

        # Load sub-skill
        try:
            sub_skill = self._skill_loader.load_skill(sub_skill_name)
        except (KeyError, Exception) as e:
            return StepResult(step_index=step.index, success=False,
                              error=f"Sub-skill '{sub_skill_name}' not found: {e}")

        logger.info("[%s] invoke_skill: calling '%s' with %d chars input",
                    skill.id, sub_skill_name, len(input_data))

        # Execute sub-workflow
        sub_ctx = dict(context)
        if input_data:
            sub_ctx["input_data"] = input_data
        sub_ctx["parent_skill"] = skill.id

        sub_result = self.execute_workflow(sub_skill, sub_ctx)

        return StepResult(
            step_index=step.index,
            success=sub_result.success,
            output=sub_result.final_output,
            error=sub_result.error,
            artifacts={
                "sub_skill": sub_skill_name,
                "sub_workflow_summary": sub_result.summary(),
                "sub_step_count": sub_result.total_steps,
            },
        )

    # ------------------------------------------------------------------
    # Action: llm_reason (default)
    # ------------------------------------------------------------------

    def _action_llm_reason(self, step: SkillStep, skill: Skill,
                           context: dict[str, Any], prev: dict[int, StepResult]) -> StepResult:
        if not self._llm_callback:
            return StepResult(step_index=step.index, success=True,
                              output=f"[LLM reasoning needed] Step {step.index}: {step.description}")

        prev_context = ""
        for idx, res in sorted(prev.items()):
            if res.success and res.output:
                # Truncate long outputs to save tokens
                out = res.output[:2000] + "..." if len(res.output) > 2000 else res.output
                prev_context += f"\n--- Step {idx} output ---\n{out}\n"

        system_prompt = (
            f"You are executing skill workflow: {skill.metadata.name}\n"
            f"Description: {skill.metadata.description}\n\n"
            f"Instructions:\n{skill.body}\n\n"
            f"Previous outputs:{prev_context}\n"
            f"Context: {context}\n"
        )
        user_message = f"Execute step {step.index}: {step.description}\nProvide the output."

        try:
            output = self._llm_callback(system_prompt, user_message)
            return StepResult(step_index=step.index, success=True, output=output)
        except Exception as e:
            return StepResult(step_index=step.index, success=False, error=f"LLM call failed: {e}")
