#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# Запуск на сервере с GPU (P100, Ubuntu)
# Один раз запустил — отключился, потом подключаешься и смотришь.
# ═══════════════════════════════════════════════════════════════════════
set -e

echo "=== Шаг 1: ставим uv (если нет) ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "=== Шаг 2: патчим pyproject.toml под CUDA ==="
# Меняем CPU-индекс PyTorch на CUDA
sed -i 's|https://download.pytorch.org/whl/cpu|https://download.pytorch.org/whl/cu124|' pyproject.toml

echo "=== Шаг 3: создаём venv и ставим зависимости ==="
uv venv
source .venv/bin/activate
uv pip install -e .

echo "=== Шаг 4: проверяем GPU ==="
python -c "
import torch
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'Доступна: {torch.cuda.is_available()}')
print(f'Устройств: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU:      {torch.cuda.get_device_name(0)}')
"

echo "=== Шаг 5: запускаем в screen ==="
SCREEN_NAME="xor_tune"

# Убиваем старую сессию если есть
screen -S "$SCREEN_NAME" -X quit 2>/dev/null || true

screen -dmS "$SCREEN_NAME" bash -c "
    source .venv/bin/activate
    python tune.py 2>&1 | tee tune_$(date +%Y%m%d_%H%M%S).log
"

echo ""
echo "══════════════════════════════════════════════"
echo "  ГОТОВО! Обучение запущено в фоне."
echo ""
echo "  Подсмотреть в реальном времени:"
echo "    ssh-подключись → screen -r $SCREEN_NAME"
echo ""
echo "  Отключиться от screen (оставив работать):"
echo "    Ctrl+A, затем D"
echo ""
echo "  Посмотреть хвост лога:"
echo "    tail -f tune_*.log"
echo ""
echo "  Убить если что:"
echo "    screen -S $SCREEN_NAME -X quit"
echo "══════════════════════════════════════════════"
