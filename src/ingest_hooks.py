from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from src.sources.base import FetchedDocument, SourceDocument

# The two points, named once so a manifest and this module cannot disagree.
HOOK_BEFORE_SEND = "before_send"
HOOK_AFTER_READ = "after_read"

HOOK_POINTS: tuple[str, ...] = (HOOK_BEFORE_SEND, HOOK_AFTER_READ)

#: Transform the fetched bytes before they leave the network.
BeforeSend = Callable[[SourceDocument, FetchedDocument], FetchedDocument]
#: Transform the reader's kept output before pieces are cut from it.
AfterRead = Callable[[str, dict], dict]


class HookError(Exception):
    pass


# Every hook this build has, by the name a manifest may use. Empty: the first
# client needs none, and an entry here is a code change on purpose (P-011).
REGISTERED_HOOKS: Mapping[str, tuple[str, Callable]] = {}


@dataclass(frozen=True, slots=True)
class StageHooks:
    before_send: tuple[BeforeSend, ...] = ()
    after_read: tuple[AfterRead, ...] = ()

    @property
    def empty(self) -> bool:
        return not self.before_send and not self.after_read

    def apply_before_send(
        self, document: SourceDocument, fetched: FetchedDocument
    ) -> FetchedDocument:
        for hook in self.before_send:
            fetched = hook(document, fetched)
            if not isinstance(fetched, FetchedDocument):
                raise HookError(
                    f"a {HOOK_BEFORE_SEND} hook returned {type(fetched).__name__} instead of "
                    "the document it was given; a hook transforms its input and returns it, "
                    "so that a caller can tell whether anything happened"
                )
        return fetched

    def apply_after_read(self, external_id: str, payload: dict) -> dict:
        for hook in self.after_read:
            payload = hook(external_id, payload)
            if not isinstance(payload, dict):
                raise HookError(
                    f"an {HOOK_AFTER_READ} hook returned {type(payload).__name__} instead of "
                    "the reader's output; the pieces are cut from what it returns"
                )
        return payload


#: What a deployment with no client-specific module runs. This one.
NO_HOOKS = StageHooks()


def hooks_for(names: Sequence[str]) -> StageHooks:
    before: list[BeforeSend] = []
    after: list[AfterRead] = []
    unknown = [name for name in names if name not in REGISTERED_HOOKS]
    if unknown:
        available = ", ".join(sorted(REGISTERED_HOOKS)) or "none — this build registers no hooks"
        raise HookError(
            f"no hook is registered under {', '.join(repr(name) for name in unknown)}; "
            f"registered hooks are {available}. A hook is reached by a name that is data and "
            "is code that was reviewed (principle P-011), so adding one is an edit here"
        )
    for name in names:
        point, hook = REGISTERED_HOOKS[name]
        if point == HOOK_BEFORE_SEND:
            before.append(hook)
        else:
            after.append(hook)
    return StageHooks(before_send=tuple(before), after_read=tuple(after))
