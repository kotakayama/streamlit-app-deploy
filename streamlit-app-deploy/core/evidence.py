from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import pandas as pd


@dataclass
class EvidenceItem:
    field_name: str
    value: float | str | None
    source_type: str  # internal_pdf / manual / external / calc / llm
    page: int | None = None
    raw_label: str | None = None
    unit: str | None = None
    as_of: str | None = None
    source_url: str | None = None
    retrieved_at: str | None = None
    calc_formula: str | None = None
    notes: str | None = None


class EvidenceLog:
    def __init__(self):
        self.items: list[EvidenceItem] = []
        self.urls: list[dict] = []

    def add(self, field_name, value, source_type, page=None, raw_label=None,
            unit=None, as_of=None, source_url=None, retrieved_at=None,
            calc_formula=None, notes=None):
        if retrieved_at is None:
            retrieved_at = datetime.now(timezone.utc).isoformat()
        self.items.append(EvidenceItem(
            field_name=field_name,
            value=value,
            source_type=source_type,
            page=page,
            raw_label=raw_label,
            unit=unit,
            as_of=as_of,
            source_url=source_url,
            retrieved_at=retrieved_at,
            calc_formula=calc_formula,
            notes=notes
        ))

    def add_url(self, url: str):
        self.urls.append({"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat()})

    def to_df(self):
        return pd.DataFrame([asdict(x) for x in self.items])

    def to_dict(self):
        return {"items": [asdict(x) for x in self.items], "urls": self.urls}