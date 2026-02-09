"""
Skill Loader - discovers, loads, and manages skills from the filesystem.

Supports:
  - Multiple directory scanning for SKILL.md files
  - Single skill loading by path (file or directory)
  - Progressive disclosure (frontmatter only → full content)
  - Reference file and script loading
  - Hot-reload capability
  - Mixed loading: directories + individual skills + inline content
"""

from __future__ import annotations

import logging
from pathlib import Path

from .models import (
    Skill,
    SkillMetadata,
    SkillStatus,
    parse_frontmatter,
    parse_workflow_steps,
)

logger = logging.getLogger(__name__)


class SkillLoader:
    """Discovers and loads skills from filesystem directories and individual paths.

    Supports three loading modes:
      1. Directory scanning: `add_skill_dir()` + `discover()`
      2. Single skill loading: `add_skill_path()` for individual SKILL.md or skill folder
      3. Inline content: `load_from_content()` for programmatic skill injection

    Example:
        loader = SkillLoader()

        # Mode 1: Multiple directories
        loader.add_skill_dir("./skills/public/")
        loader.add_skill_dir("./skills/user/")
        loader.add_skill_dir("./skills/example/")
        loader.discover()

        # Mode 2: Single skill by path
        loader.add_skill_path("./my_project/custom_skill/SKILL.md")
        loader.add_skill_path("./my_project/another_skill/")  # dir with SKILL.md

        # Mode 3: Inline
        loader.load_from_content("inline-skill", "---\\nname: x\\n---\\n# body")

        # Mix freely - all three modes register into the same skill registry
        skill = loader.load_skill("custom_skill")  # fully load by name
    """

    def __init__(
        self,
        skill_dirs: list[str | Path] | None = None,
        skill_paths: list[str | Path] | None = None,
    ):
        self._skill_dirs: list[Path] = []
        self._skills: dict[str, Skill] = {}  # name → Skill

        # Register directories
        if skill_dirs:
            for d in skill_dirs:
                self.add_skill_dir(d)

        # Register individual skill paths
        if skill_paths:
            for p in skill_paths:
                self.add_skill_path(p)

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def add_skill_dir(self, path: str | Path) -> None:
        """Register a directory to scan for skills.

        The directory will be recursively scanned for SKILL.md files
        during `discover()`. Supports adding multiple directories.

        Args:
            path: Directory path. Can be absolute or relative.
                  Non-existent paths are silently ignored with a warning.
        """
        p = Path(path).resolve()
        if not p.exists():
            logger.warning("Skill directory does not exist, skipping: %s", p)
            return
        if not p.is_dir():
            logger.warning(
                "Path is not a directory, use add_skill_path() instead: %s", p
            )
            return
        if p not in self._skill_dirs:
            self._skill_dirs.append(p)
            logger.info("Registered skill directory: %s", p)

    # ------------------------------------------------------------------
    # Single skill path loading
    # ------------------------------------------------------------------

    def add_skill_path(self, path: str | Path, auto_load: bool = False) -> Skill | None:
        """Register and discover a single skill by its path.

        Accepts either:
          - A SKILL.md file path directly
          - A directory containing a SKILL.md file

        Args:
            path: Path to a SKILL.md file or its parent directory.
            auto_load: If True, fully load the skill immediately
                       (body + references + scripts). Default is False
                       (frontmatter only, progressive disclosure).

        Returns:
            The discovered (or loaded) Skill, or None if invalid.

        Example:
            # All three forms work:
            loader.add_skill_path("./skills/event-doc-parser/SKILL.md")
            loader.add_skill_path("./skills/event-doc-parser/")
            loader.add_skill_path("./skills/event-doc-parser")
        """
        p = Path(path).resolve()

        # Resolve to SKILL.md
        skill_md = self._resolve_skill_md(p)
        if not skill_md:
            logger.warning(
                "Cannot find SKILL.md at path: %s. "
                "Provide either a SKILL.md file or a directory containing one.",
                path,
            )
            return None

        # Discover (frontmatter only)
        try:
            skill = self._discover_one(skill_md)
        except Exception as e:
            logger.warning("Failed to discover skill at %s: %s", skill_md, e)
            return None

        if not skill:
            return None

        # Check for duplicate
        if skill.id in self._skills:
            existing = self._skills[skill.id]
            if existing.path != skill_md:
                logger.warning(
                    "Skill '%s' already registered from %s, overwriting with %s",
                    skill.id,
                    existing.path,
                    skill_md,
                )
            else:
                logger.debug("Skill '%s' already registered, skipping", skill.id)
                return existing

        self._skills[skill.id] = skill
        logger.info("Added single skill: %s from %s", skill.id, skill_md)

        # Optionally fully load
        if auto_load:
            return self.load_skill(skill.id)

        return skill

    def _resolve_skill_md(self, path: Path) -> Path | None:
        """Resolve a path to a SKILL.md file.

        Handles:
          - Direct SKILL.md file path
          - Directory containing SKILL.md
          - Directory with nested skill dirs (returns None, use add_skill_dir)
        """
        if path.is_file():
            if path.name == "SKILL.md":
                return path
            # Accept any .md file as a skill definition
            if path.suffix == ".md":
                return path
            logger.warning("File is not a SKILL.md: %s", path)
            return None

        if path.is_dir():
            candidate = path / "SKILL.md"
            if candidate.exists():
                return candidate
            # Try case-insensitive
            for f in path.iterdir():
                if f.name.lower() == "skill.md" and f.is_file():
                    return f
            logger.warning("No SKILL.md found in directory: %s", path)
            return None

        return None

    # ------------------------------------------------------------------
    # Batch loading: discover from all registered directories
    # ------------------------------------------------------------------

    def discover(self) -> list[Skill]:
        """Scan all registered directories for SKILL.md files.

        Only loads frontmatter (lightweight / progressive disclosure).
        Call `load_skill(name)` to read the full body and references.

        Returns:
            List of newly discovered skills.
        """
        discovered = []
        for skill_dir in self._skill_dirs:
            for skill_md in skill_dir.rglob("SKILL.md"):
                try:
                    skill = self._discover_one(skill_md)
                    if skill and skill.id not in self._skills:
                        self._skills[skill.id] = skill
                        discovered.append(skill)
                        logger.info("Discovered skill: %s at %s", skill.id, skill_md)
                except Exception as e:
                    logger.warning("Failed to discover skill at %s: %s", skill_md, e)

        return discovered

    def discover_and_load_all(self) -> list[Skill]:
        """Discover all skills AND fully load them immediately.

        Useful when you want all skills ready upfront (small skill sets).

        Returns:
            List of fully loaded skills.
        """
        discovered = self.discover()
        loaded = []
        for skill in discovered:
            try:
                self.load_skill(skill.id)
                loaded.append(skill)
            except Exception as e:
                logger.warning("Failed to load skill '%s': %s", skill.id, e)
        return loaded

    # ------------------------------------------------------------------
    # Internal discovery
    # ------------------------------------------------------------------

    def _discover_one(self, skill_md: Path) -> Skill | None:
        """Parse frontmatter only (progressive disclosure)."""
        content = skill_md.read_text(encoding="utf-8")
        fm, body_preview = parse_frontmatter(content)

        if "name" not in fm:
            # Infer name from directory
            fm["name"] = skill_md.parent.name

        metadata = SkillMetadata(
            name=fm.get("name", "unknown"),
            description=fm.get("description", ""),
            version=fm.get("version", "1.0.0"),
            tags=fm.get("tags", []),
            inputs=fm.get("inputs", []),
            outputs=fm.get("outputs", []),
            depends_on_skills=fm.get("depends_on_skills", []),
        )

        return Skill(
            metadata=metadata,
            body="",  # Not loaded yet
            path=skill_md,
            status=SkillStatus.DISCOVERED,
        )

    # ------------------------------------------------------------------
    # Full loading
    # ------------------------------------------------------------------

    def load_skill(self, name_or_path: str) -> Skill:
        """Fully load a skill by name or path.

        Loads body, workflow steps, references, and scripts.

        Args:
            name_or_path: Either:
              - A skill name (must be already discovered/registered)
              - A filesystem path to a SKILL.md or skill directory
                (will be auto-registered if not yet known)

        Returns:
            The fully loaded Skill.

        Raises:
            KeyError: If skill name not found and path doesn't exist.
            FileNotFoundError: If the SKILL.md file is missing.
        """
        # First try as a registered skill name
        skill = self._skills.get(name_or_path)

        # If not found by name, try as a filesystem path
        if skill is None:
            candidate = Path(name_or_path)
            if candidate.exists():
                skill = self.add_skill_path(candidate)
            if skill is None:
                available = list(self._skills.keys())
                raise KeyError(
                    f"Skill '{name_or_path}' not found. "
                    f"Available: {available}. "
                    f"Or provide a valid filesystem path."
                )

        # Already fully loaded?
        if skill.status in (SkillStatus.LOADED, SkillStatus.ACTIVE):
            return skill

        if not skill.path or not skill.path.exists():
            raise FileNotFoundError(f"Skill file not found: {skill.path}")

        # Read full content
        content = skill.path.read_text(encoding="utf-8")
        _, body = parse_frontmatter(content)
        skill.body = body

        # Parse workflow steps
        skill.workflow_steps = parse_workflow_steps(body)

        # Load reference files
        skill.reference_files = self._load_references(skill.path.parent)

        # Load scripts
        skill.scripts = self._load_scripts(skill.path.parent)

        skill.status = SkillStatus.LOADED
        logger.info(
            "Loaded skill '%s': %d workflow steps, %d references, %d scripts",
            skill.id,
            len(skill.workflow_steps),
            len(skill.reference_files),
            len(skill.scripts),
        )
        return skill

    def reload_skill(self, name: str) -> Skill:
        """Force reload a skill from disk (hot-reload).

        Re-reads the SKILL.md and all associated files.

        Args:
            name: Skill name to reload.

        Returns:
            The reloaded Skill.
        """
        skill = self._skills.get(name)
        if not skill:
            raise KeyError(f"Skill '{name}' not found for reload.")

        # Reset state to trigger full reload
        skill.status = SkillStatus.DISCOVERED
        skill.body = ""
        skill.workflow_steps = []
        skill.reference_files = {}
        skill.scripts = {}

        return self.load_skill(name)

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    def _load_references(self, skill_dir: Path) -> dict[str, str]:
        """Load all reference files under references/."""
        refs = {}
        ref_dir = skill_dir / "references"
        if ref_dir.is_dir():
            for f in ref_dir.rglob("*"):
                if f.is_file() and f.suffix in (
                    ".md",
                    ".txt",
                    ".json",
                    ".yaml",
                    ".yml",
                ):
                    key = str(f.relative_to(ref_dir))
                    try:
                        refs[key] = f.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning("Failed to read reference %s: %s", f, e)
        return refs

    def _load_scripts(self, skill_dir: Path) -> dict[str, str]:
        """Load all script files under scripts/."""
        scripts = {}
        script_dir = skill_dir / "scripts"
        if script_dir.is_dir():
            for f in script_dir.rglob("*"):
                if f.is_file() and f.suffix in (".py", ".sh", ".js"):
                    key = str(f.relative_to(script_dir))
                    try:
                        scripts[key] = f.read_text(encoding="utf-8")
                    except Exception as e:
                        logger.warning("Failed to read script %s: %s", f, e)
        return scripts

    # ------------------------------------------------------------------
    # Inline content loading
    # ------------------------------------------------------------------

    def load_from_content(
        self,
        name: str,
        content: str,
        references: dict[str, str] | None = None,
        scripts: dict[str, str] | None = None,
    ) -> Skill:
        """Load a skill from inline content (not from filesystem).

        Args:
            name: Fallback name if not specified in frontmatter.
            content: Full SKILL.md content (YAML frontmatter + markdown body).
            references: Optional dict of {filename: content} for reference files.
            scripts: Optional dict of {filename: content} for scripts.

        Returns:
            The loaded Skill.
        """
        fm, body = parse_frontmatter(content)
        fm.setdefault("name", name)

        metadata = SkillMetadata(
            name=fm.get("name", name),
            description=fm.get("description", ""),
            version=fm.get("version", "1.0.0"),
            tags=fm.get("tags", []),
            inputs=fm.get("inputs", []),
            outputs=fm.get("outputs", []),
            depends_on_skills=fm.get("depends_on_skills", []),
        )

        skill = Skill(
            metadata=metadata,
            body=body,
            path=None,
            status=SkillStatus.LOADED,
            workflow_steps=parse_workflow_steps(body),
            reference_files=references or {},
            scripts=scripts or {},
        )

        self._skills[skill.id] = skill
        return skill

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def remove_skill(self, name: str) -> bool:
        """Remove a skill from the registry.

        Args:
            name: Skill name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._skills:
            del self._skills[name]
            logger.info("Removed skill: %s", name)
            return True
        return False

    def clear(self) -> None:
        """Remove all skills and directories."""
        self._skills.clear()
        self._skill_dirs.clear()
        logger.info("Cleared all skills and directories")

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name (may not be fully loaded)."""
        return self._skills.get(name)

    def has_skill(self, name: str) -> bool:
        """Check if a skill is registered."""
        return name in self._skills

    def list_skills(self) -> list[Skill]:
        """List all registered skills."""
        return list(self._skills.values())

    def list_skill_names(self) -> list[str]:
        """List all registered skill names."""
        return list(self._skills.keys())

    def list_summaries(self) -> str:
        """Generate a compact summary of all skills for LLM context."""
        if not self._skills:
            return "No skills available."

        lines = ["Available Skills:"]
        for skill in self._skills.values():
            lines.append(f"  - {skill.summary()}")
        return "\n".join(lines)

    def list_dirs(self) -> list[Path]:
        """List all registered skill directories."""
        return list(self._skill_dirs)

    def match_skill(self, query: str) -> Skill | None:
        """Simple keyword matching to find a relevant skill."""
        query_lower = query.lower()
        best_score = 0
        best_skill = None

        for skill in self._skills.values():
            score = 0
            if skill.metadata.name.lower() in query_lower:
                score += 10
            desc_words = skill.metadata.description.lower().split()
            for w in desc_words:
                if len(w) > 2 and w in query_lower:
                    score += 1
            for tag in skill.metadata.tags:
                if tag.lower() in query_lower:
                    score += 5

            if score > best_score:
                best_score = score
                best_skill = skill

        return best_skill if best_score > 0 else None

    def match_skills(self, query: str, top_k: int = 3) -> list[tuple[Skill, int]]:
        """Match multiple skills by relevance score.

        Args:
            query: Natural language query.
            top_k: Maximum number of results.

        Returns:
            List of (Skill, score) tuples, sorted by score descending.
        """
        query_lower = query.lower()
        scored = []

        for skill in self._skills.values():
            score = 0
            if skill.metadata.name.lower() in query_lower:
                score += 10
            desc_words = skill.metadata.description.lower().split()
            for w in desc_words:
                if len(w) > 2 and w in query_lower:
                    score += 1
            for tag in skill.metadata.tags:
                if tag.lower() in query_lower:
                    score += 5
            if score > 0:
                scored.append((skill, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def get_skills_context(self, max_tokens: int = 2000) -> str:
        """Build a context string for the LLM, respecting token budget.

        Progressive disclosure: only include summaries until a skill is
        explicitly activated via read_skill.
        """
        lines = [
            "<available_skills>",
            "The following skills are available. To use a skill, call the "
            "`read_skill` tool with the skill name to load its full instructions.",
            "",
        ]
        for skill in self._skills.values():
            lines.append(f"- **{skill.metadata.name}**: {skill.metadata.description}")
            if skill.metadata.tags:
                lines.append(f"  Tags: {', '.join(skill.metadata.tags)}")

        lines.append("</available_skills>")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills

    def __getitem__(self, name: str) -> Skill:
        skill = self._skills.get(name)
        if not skill:
            raise KeyError(f"Skill '{name}' not found")
        return skill

    def __repr__(self) -> str:
        return (
            f"SkillLoader(dirs={len(self._skill_dirs)}, "
            f"skills={len(self._skills)}: {list(self._skills.keys())})"
        )
