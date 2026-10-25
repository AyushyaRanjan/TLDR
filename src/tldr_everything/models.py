"""Typed results returned by the pipeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class WarningFlag(BaseModel):
    """A check the model output failed, with the text still returned."""

    code: str
    message: str


class ProgressUpdate(BaseModel):
    """One progress event for the CLI and the Streamlit UI."""

    done: int
    total: int
    label: str
    eta_seconds: float | None = None


class DocumentStats(BaseModel):
    """Size and reading-time figures for the source document."""

    source_name: str
    pages: int
    words: int
    chunks: int
    reading_minutes: float
    summary_words: int = 0
    summary_minutes: float = 0


class LayerResult(BaseModel):
    """One generated section, plus any warning flags attached to it."""

    text: str
    warnings: list[WarningFlag] = Field(default_factory=list)


class StandardResult(BaseModel):
    """Standard mode: up to three layers. Missing layers are ``None``."""

    mode: str = "standard"
    summary_30s: LayerResult | None = None
    explain_3m: LayerResult | None = None
    notes: LayerResult | None = None
    stats: DocumentStats
    single_pass: bool = False

    def all_warnings(self) -> list[WarningFlag]:
        flags: list[WarningFlag] = []
        for layer in (self.summary_30s, self.explain_3m, self.notes):
            if layer is not None:
                flags.extend(layer.warnings)
        return flags


class ClosestSection(BaseModel):
    """A chunk shown when the document does not cover the user's goal."""

    start_page: int
    end_page: int
    score: int
    excerpt: str


class GoalResult(BaseModel):
    """Goal mode sections. Empty when the document does not cover the goal."""

    mode: str = "goal"
    goal: str
    covered: bool
    message: str = ""
    prerequisites: LayerResult = Field(default_factory=lambda: LayerResult(text=""))
    steps: LayerResult = Field(default_factory=lambda: LayerResult(text=""))
    relevant_parts: LayerResult = Field(default_factory=lambda: LayerResult(text=""))
    skip: LayerResult = Field(default_factory=lambda: LayerResult(text=""))
    gotchas: LayerResult = Field(default_factory=lambda: LayerResult(text=""))
    closest: list[ClosestSection] = Field(default_factory=list)
    stats: DocumentStats

    def all_warnings(self) -> list[WarningFlag]:
        flags: list[WarningFlag] = []
        for layer in (
            self.prerequisites,
            self.steps,
            self.relevant_parts,
            self.skip,
            self.gotchas,
        ):
            flags.extend(layer.warnings)
        return flags


Result = StandardResult | GoalResult
