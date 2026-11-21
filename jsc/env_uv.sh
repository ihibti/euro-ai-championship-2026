# Source on a JURECA login node or inside a job (bash):  source jsc/env_uv.sh
# Follows the organizer slides: uv-managed Python/venv, uv caches under $PROJECT/$USER.
# Requires an activated project (jutil env activate -p <project>) so $PROJECT/$SCRATCH exist.

if [ -z "${PROJECT:-}" ]; then
  echo "env_uv.sh: \$PROJECT is not set; run 'jutil env activate -p <project>' first" >&2
  return 1 2>/dev/null || exit 1
fi

EUROAI_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export EUROAI_REPO
export UV_PYTHON_INSTALL_DIR="${PROJECT}/${USER}/uv/python"
export UV_CACHE_DIR="${PROJECT}/${USER}/uv/cache"
export UV_TOOL_DIR="${PROJECT}/${USER}/uv/tools"

if [ -n "${SCRATCH:-}" ]; then
  export HF_HOME="${SCRATCH}/${USER}/hf"
  export TORCH_HOME="${SCRATCH}/${USER}/torch"
  mkdir -p "${HF_HOME}" "${TORCH_HOME}"
fi

if [ -f "${EUROAI_REPO}/.venv/bin/activate" ]; then
  source "${EUROAI_REPO}/.venv/bin/activate"
else
  echo "env_uv.sh: no .venv yet; on a login node run: cd ${EUROAI_REPO} && uv sync --extra dev --extra torch" >&2
fi
