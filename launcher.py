from __future__ import annotations

import os
import sys
import traceback


def _show_error(message: str) -> None:
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "Video Editor", 0x10)
    except Exception:
        pass


def main() -> int:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    src_dir = os.path.join(base_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from main import main as app_main

        app_main()
        return 0
    except Exception:
        log_path = os.path.join(base_dir, "launch_error.log")
        with open(log_path, "w", encoding="utf-8") as log_file:
            traceback.print_exc(file=log_file)

        _show_error(
            "Program acilamadi.\n\n"
            "Detaylar launch_error.log dosyasina yazildi.\n\n"
            f"Dosya: {log_path}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
