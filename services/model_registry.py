from dataclasses import asdict, dataclass
from typing import Literal

ModelLanguageScope = Literal["multilingual", "english"]
ModelFamily = Literal["standard", "distil"]


@dataclass(frozen=True)
class ModelRegistryEntry:
    id: str
    displayName: str
    backendValue: str
    hfRepoId: str
    languageScope: ModelLanguageScope
    family: ModelFamily
    enabled: bool = True


MODEL_REGISTRY: tuple[ModelRegistryEntry, ...] = (
    ModelRegistryEntry(
        id="tiny",
        displayName="Tiny",
        backendValue="tiny",
        hfRepoId="Systran/faster-whisper-tiny",
        languageScope="multilingual",
        family="standard",
    ),
    ModelRegistryEntry(
        id="base",
        displayName="Base",
        backendValue="base",
        hfRepoId="Systran/faster-whisper-base",
        languageScope="multilingual",
        family="standard",
    ),
    ModelRegistryEntry(
        id="small",
        displayName="Small",
        backendValue="small",
        hfRepoId="Systran/faster-whisper-small",
        languageScope="multilingual",
        family="standard",
    ),
    ModelRegistryEntry(
        id="medium",
        displayName="Medium",
        backendValue="medium",
        hfRepoId="Systran/faster-whisper-medium",
        languageScope="multilingual",
        family="standard",
    ),
    ModelRegistryEntry(
        id="large-v1",
        displayName="Large V1",
        backendValue="large-v1",
        hfRepoId="Systran/faster-whisper-large-v1",
        languageScope="multilingual",
        family="standard",
    ),
    ModelRegistryEntry(
        id="large-v2",
        displayName="Large V2",
        backendValue="large-v2",
        hfRepoId="Systran/faster-whisper-large-v2",
        languageScope="multilingual",
        family="standard",
    ),
    ModelRegistryEntry(
        id="large-v3",
        displayName="Large V3",
        backendValue="large-v3",
        hfRepoId="Systran/faster-whisper-large-v3",
        languageScope="multilingual",
        family="standard",
    ),
    # The faster-whisper backend resolves "large" to the large-v3 repository.
    ModelRegistryEntry(
        id="large",
        displayName="Large",
        backendValue="large",
        hfRepoId="Systran/faster-whisper-large-v3",
        languageScope="multilingual",
        family="standard",
    ),
    ModelRegistryEntry(
        id="distil-large-v2",
        displayName="Distil Large V2",
        backendValue="distil-large-v2",
        hfRepoId="Systran/faster-distil-whisper-large-v2",
        languageScope="multilingual",
        family="distil",
    ),
    ModelRegistryEntry(
        id="distil-medium.en",
        displayName="Distil Medium English",
        backendValue="distil-medium.en",
        hfRepoId="Systran/faster-distil-whisper-medium.en",
        languageScope="english",
        family="distil",
    ),
    ModelRegistryEntry(
        id="distil-small.en",
        displayName="Distil Small English",
        backendValue="distil-small.en",
        hfRepoId="Systran/faster-distil-whisper-small.en",
        languageScope="english",
        family="distil",
    ),
    ModelRegistryEntry(
        id="tiny.en",
        displayName="Tiny English",
        backendValue="tiny.en",
        hfRepoId="Systran/faster-whisper-tiny.en",
        languageScope="english",
        family="standard",
    ),
    ModelRegistryEntry(
        id="base.en",
        displayName="Base English",
        backendValue="base.en",
        hfRepoId="Systran/faster-whisper-base.en",
        languageScope="english",
        family="standard",
    ),
    ModelRegistryEntry(
        id="small.en",
        displayName="Small English",
        backendValue="small.en",
        hfRepoId="Systran/faster-whisper-small.en",
        languageScope="english",
        family="standard",
    ),
    ModelRegistryEntry(
        id="medium.en",
        displayName="Medium English",
        backendValue="medium.en",
        hfRepoId="Systran/faster-whisper-medium.en",
        languageScope="english",
        family="standard",
    ),
)


def get_model_registry() -> tuple[ModelRegistryEntry, ...]:
    return MODEL_REGISTRY


def list_model_catalog() -> list[dict[str, str | bool]]:
    return [asdict(entry) for entry in MODEL_REGISTRY]


def list_enabled_backend_values() -> list[str]:
    return [entry.backendValue for entry in MODEL_REGISTRY if entry.enabled]
