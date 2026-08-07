from enum import Enum


class EnvironmentType(str, Enum):
    DOCKER = "docker"
    DAYTONA = "daytona"
    E2B = "e2b"
    MODAL = "modal"
    RUNLOOP = "runloop"
    GKE = "gke"
    APPLE_CONTAINER = "apple-container"
    # Local rootless-podman sandbox (SWE-Together on-cluster harness). Selected via
    # EnvironmentConfig.import_path (see src/podman_env.py), so it is intentionally
    # NOT registered in EnvironmentFactory._ENVIRONMENT_MAP; this value only makes
    # PodmanEnvironment.type() and logs honest.
    PODMAN = "podman"
    # Remote Sandoq OCI-run sandbox. Harbor resolves it lazily through the
    # EnvironmentFactory import-path map so the standalone Harbor package does
    # not need to import SWE-Together's optional backend at module import time.
    SANDOQ = "sandoq"
