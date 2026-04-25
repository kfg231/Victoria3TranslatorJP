"""Tkinter-based GUI for the Victoria 3 translator."""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

from . import logger as log_setup
from .cache_manager import TranslationCache
from .deepseek_client import (
    DeepSeekClient,
    DEFAULT_BASE_URL,
    OPENROUTER_BASE_URL,
    normalize_model_name,
)
from .language_config import LANGUAGES, list_display_names
from .openrouter_client import (
    fetch_available_models,
    format_model_label,
    search_models,
)
from .translator import TranslationResult, translate_mod


logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"
CACHE_PATH = PROJECT_ROOT / "cache" / "translation_cache.sqlite"
LOG_PATH = PROJECT_ROOT / "logs" / "translator.log"
DEFAULT_SOURCE = PROJECT_ROOT / "localization"
DEFAULT_OUTPUT = PROJECT_ROOT / "output"


DEEPSEEK_MODELS = [
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-reasoner",
]


class ModelSearchDialog:
    """Modal dialog to search & select an OpenRouter model."""

    def __init__(self, parent: tk.Misc, api_key: str):
        self.parent = parent
        self.api_key = api_key
        self.result: Optional[str] = None  # selected model id

        self._models: List[Dict[str, Any]] = []
        self._filtered: List[Dict[str, Any]] = []

        # Build window
        self.win = tk.Toplevel(parent)
        self.win.title("OpenRouter モデル検索")
        self.win.geometry("720x520")
        self.win.minsize(500, 300)
        self.win.transient(parent)  # type: ignore[arg-type]
        self.win.grab_set()

        self._build_ui()
        self._fetch_models()

        self.win.wait_window()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        # status
        self.var_status = tk.StringVar(value="モデル一覧を取得中...")
        ttk.Label(self.win, textvariable=self.var_status, font=("", 9, "italic")).pack(
            anchor="w", padx=6, pady=3
        )

        # search
        search_frame = ttk.Frame(self.win)
        search_frame.pack(fill="x", padx=6, pady=3)

        ttk.Label(search_frame, text="検索:").pack(side="left")
        self.var_search = tk.StringVar()
        search_entry = ttk.Entry(search_frame, textvariable=self.var_search, width=40)
        search_entry.pack(side="left", padx=4)
        search_entry.bind("<KeyRelease>", self._on_search)
        ttk.Button(search_frame, text="クリア", command=self._clear_search).pack(
            side="left", padx=4
        )

        # listbox + scrollbar
        list_frame = ttk.Frame(self.win)
        list_frame.pack(fill="both", expand=True, padx=6, pady=3)

        self.listbox = tk.Listbox(
            list_frame, font=("Consolas", 9), exportselection=False
        )
        vsb = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=vsb.set)
        self.listbox.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.listbox.bind("<Double-Button-1>", self._on_select)

        # buttons
        btn_frame = ttk.Frame(self.win)
        btn_frame.pack(fill="x", padx=6, pady=3)

        ttk.Button(btn_frame, text="選択", command=self._on_select).pack(
            side="right", padx=4
        )
        ttk.Button(btn_frame, text="キャンセル", command=self.win.destroy).pack(
            side="right", padx=4
        )

    # --------------------------------------------------------- data / logic
    def _fetch_models(self) -> None:
        def _work() -> None:
            try:
                models = fetch_available_models(self.api_key)
                self._models = search_models("", models)
            except Exception as e:
                self.win.after(0, lambda: self.var_status.set(f"取得失敗: {e}"))
                return
            self.win.after(0, self._populate_list)
            self.win.after(0, lambda: self.var_status.set(f"{len(self._models)} モデル"))

        threading.Thread(target=_work, daemon=True).start()

    def _populate_list(self, models: Optional[List[Dict[str, Any]]] = None) -> None:
        self.listbox.delete(0, "end")
        for m in models or self._filtered or self._models:
            label = format_model_label(m)
            self.listbox.insert("end", label)
        # Store model ids parallel to listbox indices
        self._listbox_models = models or self._filtered or self._models

    def _on_search(self, event: Optional[object] = None) -> None:
        q = self.var_search.get()
        self._filtered = search_models(q, self._models)
        self._populate_list(self._filtered)
        self.var_status.set(
            f"{len(self._filtered)} / {len(self._models)} モデル"
        )

    def _clear_search(self) -> None:
        self.var_search.set("")
        self._filtered = []
        self._populate_list(self._models)
        self.var_status.set(f"{len(self._models)} モデル")

    def _on_select(self, event: Optional[object] = None) -> None:
        sel = self.listbox.curselection()
        if not sel:
            messagebox.showwarning("選択なし", "モデルを選択してください", parent=self.win)
            return
        idx = sel[0]
        models = getattr(self, "_listbox_models", self._models)
        if idx < len(models):
            self.result = models[idx].get("id", "")
            self.win.destroy()


class TranslatorApp:
    """Main window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Victoria 3 Translator")
        self.root.geometry("820x680")
        self.root.minsize(760, 620)

        # state
        self.log_queue: queue.Queue = queue.Queue()
        self.worker_thread: Optional[threading.Thread] = None
        self.stop_flag = threading.Event()

        # settings
        self.settings = self._load_settings()
        self.cache = TranslationCache(CACHE_PATH)

        self._build_ui()
        log_setup.configure_logging(
            level=logging.INFO,
            log_file=LOG_PATH,
            gui_queue=self.log_queue,
        )
        self._drain_log_queue()

    # ============================================================= settings
    def _load_settings(self) -> dict:
        if CONFIG_PATH.exists():
            try:
                s = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                return s
            except Exception:
                return {}
        return {}

    def _save_settings(self) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(
            json.dumps(self.settings, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ================================================================== UI
    def _build_ui(self) -> None:
        # ----- frame: API / Provider
        frm_api = ttk.LabelFrame(self.root, text="API プロバイダ")
        frm_api.pack(fill="x", padx=8, pady=4)

        # Provider selector
        ttk.Label(frm_api, text="プロバイダ:").grid(
            row=0, column=0, sticky="w", padx=8, pady=4
        )
        self.var_provider = tk.StringVar(
            value=self.settings.get("provider", "deepseek")
        )
        cb_provider = ttk.Combobox(
            frm_api,
            textvariable=self.var_provider,
            values=["deepseek", "openrouter"],
            state="readonly",
            width=15,
        )
        cb_provider.grid(row=0, column=1, sticky="w", padx=8, pady=4)
        cb_provider.bind("<<ComboboxSelected>>", self._on_provider_change)

        # DeepSeek API key
        ttk.Label(frm_api, text="DeepSeek API Key:").grid(
            row=1, column=0, sticky="w", padx=8, pady=4
        )
        self.var_ds_api_key = tk.StringVar(
            value=self.settings.get("api_key_deepseek")
            or os.environ.get("DEEPSEEK_API_KEY", "")
        )
        self.entry_ds_api_key = ttk.Entry(
            frm_api, textvariable=self.var_ds_api_key, show="*", width=60
        )
        self.entry_ds_api_key.grid(row=1, column=1, sticky="we", padx=8, pady=4)

        # OpenRouter API key
        ttk.Label(frm_api, text="OpenRouter API Key:").grid(
            row=2, column=0, sticky="w", padx=8, pady=4
        )
        self.var_or_api_key = tk.StringVar(
            value=self.settings.get("api_key_openrouter")
            or os.environ.get("OPENROUTER_API_KEY", "")
        )
        self.entry_or_api_key = ttk.Entry(
            frm_api, textvariable=self.var_or_api_key, show="*", width=60
        )
        self.entry_or_api_key.grid(row=2, column=1, sticky="we", padx=8, pady=4)

        # API key visibility toggle
        ttk.Button(frm_api, text="表示/隠す", command=self._toggle_api_visibility).grid(
            row=1, column=2, rowspan=2, padx=8, pady=4
        )
        ttk.Button(frm_api, text="保存", command=self._on_save_api_key).grid(
            row=1, column=3, rowspan=2, padx=8, pady=4
        )

        # Model
        ttk.Label(frm_api, text="モデル:").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        self.var_model = tk.StringVar(
            value=self.settings.get("model", "deepseek-v4-pro")
        )
        self.cb_model = ttk.Combobox(
            frm_api,
            textvariable=self.var_model,
            values=DEEPSEEK_MODELS,
            width=40,
        )
        self.cb_model.grid(row=3, column=1, sticky="w", padx=8, pady=4)

        self.btn_search_model = ttk.Button(
            frm_api, text="モデルを検索...", command=self._on_search_model
        )
        self.btn_search_model.grid(row=3, column=2, padx=8, pady=4)

        # Thinking / Reasoning effort
        # DeepSeek direct: simple on/off checkbox
        self.var_thinking = tk.BooleanVar(
            value=bool(self.settings.get("enable_thinking", False))
        )
        self.cb_thinking = ttk.Checkbutton(
            frm_api,
            text="思考モード(高品質/高コスト)",
            variable=self.var_thinking,
        )
        self.cb_thinking.grid(row=3, column=3, columnspan=2, sticky="w", padx=8, pady=4)

        # OpenRouter: reasoning effort level selector
        self.frm_reasoning = ttk.Frame(frm_api)
        self.frm_reasoning.grid(row=4, column=0, columnspan=5, sticky="w", padx=8, pady=4)

        ttk.Label(self.frm_reasoning, text="推論レベル:").pack(side="left")
        self.var_reasoning_effort = tk.StringVar(
            value=self.settings.get("reasoning_effort", "none")
        )
        self.cb_reasoning_effort = ttk.Combobox(
            self.frm_reasoning,
            textvariable=self.var_reasoning_effort,
            values=["none", "low", "medium", "high"],
            state="readonly",
            width=10,
        )
        self.cb_reasoning_effort.pack(side="left", padx=4)
        ttk.Label(
            self.frm_reasoning,
            text="none=思考なし(高速/低コスト)  high=深い推論(高品質/遅い)",
            font=("TkDefaultFont", 8),
        ).pack(side="left", padx=4)

        # Apply provider-dependent visibility
        self._apply_provider_ui()

        frm_api.columnconfigure(1, weight=1)

        # ----- frame: languages
        frm_lang = ttk.LabelFrame(self.root, text="言語")
        frm_lang.pack(fill="x", padx=8, pady=4)

        display_pairs = list_display_names()
        self._lang_key_by_display = {display: key for key, display in display_pairs}
        display_values = [display for _, display in display_pairs]

        ttk.Label(frm_lang, text="ソース言語:").grid(row=0, column=0, sticky="w", padx=8, pady=4)
        default_src = self.settings.get("source_lang", "english")
        self.var_source_lang = tk.StringVar(value=LANGUAGES[default_src]["display"])
        ttk.Combobox(
            frm_lang,
            textvariable=self.var_source_lang,
            values=display_values,
            width=30,
            state="readonly",
        ).grid(row=0, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(frm_lang, text="ターゲット言語:").grid(
            row=0, column=2, sticky="w", padx=8, pady=4
        )
        default_tgt = self.settings.get("target_lang", "japanese")
        self.var_target_lang = tk.StringVar(value=LANGUAGES[default_tgt]["display"])
        ttk.Combobox(
            frm_lang,
            textvariable=self.var_target_lang,
            values=display_values,
            width=30,
            state="readonly",
        ).grid(row=0, column=3, sticky="w", padx=8, pady=4)

        # ----- frame: paths
        frm_paths = ttk.LabelFrame(self.root, text="フォルダ")
        frm_paths.pack(fill="x", padx=8, pady=4)

        ttk.Label(frm_paths, text="MODフォルダ:").grid(
            row=0, column=0, sticky="w", padx=8, pady=4
        )
        self.var_source_dir = tk.StringVar(
            value=self.settings.get("source_dir") or str(DEFAULT_SOURCE)
        )
        ttk.Entry(frm_paths, textvariable=self.var_source_dir).grid(
            row=0, column=1, sticky="we", padx=8, pady=4
        )
        ttk.Button(frm_paths, text="参照...", command=self._browse_source).grid(
            row=0, column=2, padx=8, pady=4
        )

        ttk.Label(frm_paths, text="出力先:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.var_output_dir = tk.StringVar(
            value=self.settings.get("output_dir") or str(DEFAULT_OUTPUT)
        )
        ttk.Entry(frm_paths, textvariable=self.var_output_dir).grid(
            row=1, column=1, sticky="we", padx=8, pady=4
        )
        ttk.Button(frm_paths, text="参照...", command=self._browse_output).grid(
            row=1, column=2, padx=8, pady=4
        )

        ttk.Label(frm_paths, text="出力フォルダ名(必須):").grid(
            row=2, column=0, sticky="w", padx=8, pady=4
        )
        self.var_output_name = tk.StringVar(
            value=self.settings.get("output_name", "")
        )
        ttk.Entry(frm_paths, textvariable=self.var_output_name).grid(
            row=2, column=1, sticky="we", padx=8, pady=4
        )

        frm_paths.columnconfigure(1, weight=1)

        # ----- frame: options
        frm_opts = ttk.LabelFrame(self.root, text="オプション")
        frm_opts.pack(fill="x", padx=8, pady=4)

        ttk.Label(frm_opts, text="MODテーマ(任意):").grid(
            row=0, column=0, sticky="w", padx=8, pady=4
        )
        self.var_mod_context = tk.StringVar(value=self.settings.get("mod_context", ""))
        ttk.Entry(frm_opts, textvariable=self.var_mod_context).grid(
            row=0, column=1, columnspan=3, sticky="we", padx=8, pady=4
        )

        ttk.Label(frm_opts, text="並列数:").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        self.var_workers = tk.IntVar(value=int(self.settings.get("max_workers", 8)))
        ttk.Spinbox(
            frm_opts, from_=1, to=999, textvariable=self.var_workers, width=6
        ).grid(row=1, column=1, sticky="w", padx=8, pady=4)

        ttk.Label(frm_opts, text="バッチサイズ:").grid(row=1, column=2, sticky="w", padx=8, pady=4)
        self.var_batch_size = tk.IntVar(value=int(self.settings.get("batch_size", 40)))
        ttk.Spinbox(
            frm_opts, from_=5, to=200, textvariable=self.var_batch_size, width=6
        ).grid(row=1, column=3, sticky="w", padx=8, pady=4)

        ttk.Label(frm_opts, text="レート制限:").grid(row=2, column=0, sticky="w", padx=8, pady=4)
        self.var_rate_limit_policy = tk.StringVar(
            value=self.settings.get("rate_limit_policy", "backoff")
        )
        policy_combo = ttk.Combobox(
            frm_opts,
            textvariable=self.var_rate_limit_policy,
            values=["backoff", "retry", "stop", "ignore"],
            state="readonly",
            width=10,
        )
        policy_combo.grid(row=2, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(frm_opts, text="(429時動作)", font=("TkDefaultFont", 8)).grid(
            row=2, column=2, sticky="w", padx=8, pady=4
        )

        ttk.Label(frm_opts, text="リトライ数:").grid(row=2, column=3, sticky="w", padx=8, pady=4)
        self.var_retries = tk.IntVar(value=int(self.settings.get("max_retries", 3)))
        ttk.Spinbox(
            frm_opts, from_=0, to=999, textvariable=self.var_retries, width=6
        ).grid(row=2, column=4, sticky="w", padx=8, pady=4)

        ttk.Label(frm_opts, text="エラー処理:").grid(row=3, column=0, sticky="w", padx=8, pady=4)
        self.var_error_policy = tk.StringVar(
            value=self.settings.get("error_policy", "retry")
        )
        error_policy_combo = ttk.Combobox(
            frm_opts,
            textvariable=self.var_error_policy,
            values=["retry", "stop", "ignore"],
            state="readonly",
            width=10,
        )
        error_policy_combo.grid(row=3, column=1, sticky="w", padx=8, pady=4)
        ttk.Label(frm_opts, text="(429以外)", font=("TkDefaultFont", 8)).grid(
            row=3, column=2, sticky="w", padx=8, pady=4
        )

        ttk.Label(frm_opts, text="エラーリトライ:").grid(row=3, column=3, sticky="w", padx=8, pady=4)
        self.var_error_retries = tk.IntVar(value=int(self.settings.get("error_retries", 1)))
        ttk.Spinbox(
            frm_opts, from_=0, to=999, textvariable=self.var_error_retries, width=6
        ).grid(row=3, column=4, sticky="w", padx=8, pady=4)

        self.var_smart_skip = tk.BooleanVar(
            value=bool(self.settings.get("smart_skip_translated", False))
        )
        ttk.Checkbutton(
            frm_opts, text="未翻訳箇所のみ抽出 (混合ファイル対応)", variable=self.var_smart_skip
        ).grid(row=4, column=0, columnspan=5, sticky="w", padx=8, pady=4)

        frm_opts.columnconfigure(1, weight=1)

        # ----- frame: actions
        frm_actions = ttk.Frame(self.root)
        frm_actions.pack(fill="x", padx=8, pady=4)

        self.btn_start = ttk.Button(
            frm_actions, text="翻訳開始", command=self._on_start
        )
        self.btn_start.pack(side="left", padx=6)
        self.btn_cancel = ttk.Button(
            frm_actions, text="キャンセル", command=self._on_cancel, state="disabled"
        )
        self.btn_cancel.pack(side="left", padx=6)
        ttk.Button(
            frm_actions, text="キャッシュをクリア", command=self._on_clear_cache
        ).pack(side="left", padx=6)
        ttk.Button(
            frm_actions, text="出力フォルダを開く", command=self._open_output
        ).pack(side="left", padx=6)

        # ----- frame: progress
        frm_prog = ttk.LabelFrame(self.root, text="進捗")
        frm_prog.pack(fill="x", padx=8, pady=4)

        self.var_progress = tk.DoubleVar(value=0.0)
        self.progress = ttk.Progressbar(
            frm_prog, variable=self.var_progress, maximum=100.0, length=400
        )
        self.progress.pack(fill="x", padx=8, pady=4)

        self.var_progress_text = tk.StringVar(value="待機中")
        ttk.Label(frm_prog, textvariable=self.var_progress_text).pack(
            anchor="w", padx=8, pady=2
        )

        # ----- frame: log
        frm_log = ttk.LabelFrame(self.root, text="ログ")
        frm_log.pack(fill="both", expand=True, padx=8, pady=4)

        self.text_log = tk.Text(
            frm_log,
            height=12,
            wrap="none",
            state="disabled",
            bg="#1e1e1e",
            fg="#e0e0e0",
        )
        vsb = ttk.Scrollbar(frm_log, orient="vertical", command=self.text_log.yview)
        self.text_log.configure(yscrollcommand=vsb.set)
        self.text_log.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        self.text_log.tag_configure("INFO", foreground="#9cdcfe")
        self.text_log.tag_configure("WARNING", foreground="#dcdcaa")
        self.text_log.tag_configure("ERROR", foreground="#f48771")
        self.text_log.tag_configure("DEBUG", foreground="#808080")

    # ------------------------------------------------------ provider UI toggle
    def _apply_provider_ui(self) -> None:
        """Show/hide widgets depending on selected provider."""
        is_deepseek = self.var_provider.get() == "deepseek"
        current_model = self.var_model.get().strip()

        # Show deepseek model combobox for DeepSeek, model search button for OpenRouter
        if is_deepseek:
            normalized_model = normalize_model_name(current_model, DEFAULT_BASE_URL)
            if normalized_model and normalized_model != current_model:
                logger.warning(
                    "Normalized UI model for DeepSeek direct API: %s -> %s",
                    current_model,
                    normalized_model,
                )
                self.var_model.set(normalized_model)
            self.cb_model.configure(values=DEEPSEEK_MODELS, state="readonly")
            self.btn_search_model.configure(state="disabled")
            # DeepSeek: show thinking checkbox, hide reasoning effort
            self.cb_thinking.grid()
            self.frm_reasoning.grid_remove()
        else:
            self.cb_model.configure(values=[], state="normal")
            self.btn_search_model.configure(state="normal")
            # OpenRouter: hide thinking checkbox, show reasoning effort
            self.cb_thinking.grid_remove()
            self.frm_reasoning.grid()

    def _on_provider_change(self, event: Optional[object] = None) -> None:
        self._apply_provider_ui()

    # =========================================================== callbacks
    def _toggle_api_visibility(self) -> None:
        current = self.entry_ds_api_key.cget("show")
        new_show = "" if current == "*" else "*"
        self.entry_ds_api_key.configure(show=new_show)
        self.entry_or_api_key.configure(show=new_show)

    def _browse_source(self) -> None:
        path = filedialog.askdirectory(
            title="MODフォルダ(localization)を選択",
            initialdir=self.var_source_dir.get() or str(DEFAULT_SOURCE),
        )
        if path:
            self.var_source_dir.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(
            title="出力フォルダを選択",
            initialdir=self.var_output_dir.get() or str(DEFAULT_OUTPUT),
        )
        if path:
            self.var_output_dir.set(path)

    def _on_save_api_key(self) -> None:
        self.settings["api_key_deepseek"] = self.var_ds_api_key.get().strip()
        self.settings["api_key_openrouter"] = self.var_or_api_key.get().strip()
        self.settings["provider"] = self.var_provider.get()
        self._save_settings()
        messagebox.showinfo("保存完了", f"APIキーを保存しました:\n{CONFIG_PATH}")

    def _on_search_model(self) -> None:
        api_key = self.var_or_api_key.get().strip()
        if not api_key:
            messagebox.showerror(
                "エラー", "OpenRouter APIキーを入力してから検索してください"
            )
            return
        dialog = ModelSearchDialog(self.root, api_key)
        if dialog.result:
            self.var_model.set(dialog.result)
            self.cb_model.configure(state="normal")

    def _on_clear_cache(self) -> None:
        if not messagebox.askyesno("確認", "翻訳キャッシュを全削除しますか？"):
            return
        deleted = self.cache.clear()
        messagebox.showinfo("完了", f"{deleted} 件のキャッシュを削除しました")

    def _open_output(self) -> None:
        path = Path(self.var_output_dir.get() or DEFAULT_OUTPUT)
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except AttributeError:
            messagebox.showinfo("出力先", str(path))

    def _on_cancel(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_flag.set()
            self.var_progress_text.set("キャンセル要求中...")
            self.btn_cancel.configure(state="disabled")

    # ------------------------------------------------------------ start/run
    def _collect_settings(self) -> Optional[dict]:
        provider = self.var_provider.get().strip()

        # Determine API key based on provider
        if provider == "deepseek":
            api_key = self.var_ds_api_key.get().strip()
            if not api_key:
                messagebox.showerror("エラー", "DeepSeek APIキーを入力してください")
                return None
        else:
            api_key = self.var_or_api_key.get().strip()
            if not api_key:
                messagebox.showerror("エラー", "OpenRouter APIキーを入力してください")
                return None

        try:
            src_key = self._lang_key_by_display[self.var_source_lang.get()]
            tgt_key = self._lang_key_by_display[self.var_target_lang.get()]
        except KeyError:
            messagebox.showerror("エラー", "言語の選択が不正です")
            return None

        if src_key == tgt_key:
            messagebox.showerror("エラー", "ソース言語とターゲット言語が同じです")
            return None

        source_dir = Path(self.var_source_dir.get().strip())
        if not source_dir.exists():
            messagebox.showerror("エラー", f"MODフォルダが存在しません:\n{source_dir}")
            return None

        output_dir = Path(self.var_output_dir.get().strip())
        output_name = self.var_output_name.get().strip()
        if not output_name:
            messagebox.showerror("エラー", "出力フォルダ名を入力してください")
            return None

        invalid_chars = '<>:"/\\|?*'
        if any(ch in output_name for ch in invalid_chars):
            messagebox.showerror(
                "エラー",
                "出力フォルダ名に使用できない文字が含まれています:\n<>:\"/\\|?*",
            )
            return None

        out_candidate = output_dir / output_name
        if out_candidate.exists():
            messagebox.showerror(
                "エラー",
                f"同名の出力フォルダが既に存在します。別名を指定してください:\n{out_candidate}",
            )
            return None

        model = self.var_model.get().strip()
        if not model:
            messagebox.showerror("エラー", "モデルを選択してください")
            return None

        base_url = OPENROUTER_BASE_URL if provider == "openrouter" else DEFAULT_BASE_URL
        normalized_model = normalize_model_name(model, base_url)
        if normalized_model != model:
            logger.warning(
                "Normalized model before translation: %s -> %s",
                model,
                normalized_model,
            )
            self.var_model.set(normalized_model)
            model = normalized_model

        batch_size = int(self.var_batch_size.get())
        initial_max_workers = int(self.var_workers.get())
        if batch_size <= 1 and initial_max_workers > 32:
            logger.warning(
                "Current settings are extremely aggressive (batch_size=%d, workers=%d). "
                "This is often slower in practice and tends to amplify API errors. "
                "Recommended starting point: batch_size 10-40, workers 4-16.",
                batch_size,
                initial_max_workers,
            )

        return {
            "provider": provider,
            "api_key": api_key,
            "source_lang_key": src_key,
            "target_lang_key": tgt_key,
            "source_root": source_dir,
            "output_root": output_dir,
            "output_name": output_name,
            "mod_context": self.var_mod_context.get().strip(),
            "model": model,
            "enable_thinking": bool(self.var_thinking.get()),
            "reasoning_effort": self.var_reasoning_effort.get(),
            "batch_size": batch_size,
            "initial_max_workers": initial_max_workers,
            "max_retries": int(self.var_retries.get()),
            "error_retries": int(self.var_error_retries.get()),
            "rate_limit_policy": self.var_rate_limit_policy.get(),
            "error_policy": self.var_error_policy.get(),
            "smart_skip_translated": bool(self.var_smart_skip.get()),
        }

    def _persist_run_settings(self, cfg: dict) -> None:
        # Always persist both API keys if they're filled
        if cfg["api_key"]:
            if cfg["provider"] == "deepseek":
                self.settings["api_key_deepseek"] = cfg["api_key"]
            else:
                self.settings["api_key_openrouter"] = cfg["api_key"]
        self.settings.update(
            {
                "provider": cfg["provider"],
                "source_lang": cfg["source_lang_key"],
                "target_lang": cfg["target_lang_key"],
                "source_dir": str(cfg["source_root"]),
                "output_dir": str(cfg["output_root"]),
                "output_name": cfg["output_name"],
                "mod_context": cfg["mod_context"],
                "model": cfg["model"],
                "enable_thinking": cfg["enable_thinking"],
                "reasoning_effort": cfg["reasoning_effort"],
                "batch_size": cfg["batch_size"],
                "max_workers": cfg["initial_max_workers"],
                "max_retries": cfg["max_retries"],
                "error_retries": cfg["error_retries"],
                "rate_limit_policy": cfg["rate_limit_policy"],
                "error_policy": cfg["error_policy"],
                "smart_skip_translated": cfg["smart_skip_translated"],
            }
        )
        self._save_settings()

    def _on_start(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("実行中", "既に翻訳タスクが進行中です")
            return

        cfg = self._collect_settings()
        if cfg is None:
            return
        self._persist_run_settings(cfg)

        self.stop_flag = threading.Event()
        self.var_progress.set(0)
        self.var_progress_text.set("開始中...")
        self.btn_start.configure(state="disabled")
        self.btn_cancel.configure(state="normal")

        self.worker_thread = threading.Thread(
            target=self._run_translation, args=(cfg,), daemon=True
        )
        self.worker_thread.start()

    def _run_translation(self, cfg: dict) -> None:
        try:
            def progress_cb(done: int, total: int, msg: str) -> None:
                pct = (done / total) * 100 if total else 0.0
                self.root.after(0, lambda: self.var_progress.set(pct))
                self.root.after(
                    0, lambda: self.var_progress_text.set(f"{pct:5.1f}% — {msg}")
                )

            reasoning_effort = cfg.get("reasoning_effort", "none")
            logger.info(
                "翻訳開始: provider=%s, %s -> %s, model=%s, thinking=%s, reasoning_effort=%s",
                cfg["provider"],
                cfg["source_lang_key"],
                cfg["target_lang_key"],
                cfg["model"],
                cfg["enable_thinking"],
                reasoning_effort,
            )

            # Determine base_url based on provider
            if cfg["provider"] == "openrouter":
                base_url = OPENROUTER_BASE_URL
            else:
                base_url = DEFAULT_BASE_URL
            enable_thinking = cfg["enable_thinking"]

            model = cfg["model"]
            # OpenRouter: if reasoning_effort is not 'none', use the
            # reasoning parameter (handled in deepseek_client._call).
            # Also support legacy :thinking suffix for backwards compat.
            if cfg["provider"] == "openrouter":
                if reasoning_effort != "none" and not model.endswith(":thinking"):
                    # Some models need :thinking suffix to enable reasoning
                    pass  # reasoning is controlled via API param, not suffix
                elif cfg["enable_thinking"] and not model.endswith(":thinking"):
                    model = model + ":thinking"

            result: TranslationResult = translate_mod(
                source_lang_key=cfg["source_lang_key"],
                target_lang_key=cfg["target_lang_key"],
                source_root=cfg["source_root"],
                output_root=cfg["output_root"],
                output_name=cfg["output_name"],
                api_key=cfg["api_key"],
                mod_context=cfg["mod_context"],
                model=model,
                enable_thinking=enable_thinking,
                base_url=base_url,
                batch_size=cfg["batch_size"],
                initial_max_workers=cfg["initial_max_workers"],
                max_retries=cfg["max_retries"],
                error_retries=cfg["error_retries"],
                rate_limit_policy=cfg["rate_limit_policy"],
                error_policy=cfg["error_policy"],
                cache=self.cache,
                progress_cb=progress_cb,
                stop_flag=self.stop_flag,
                smart_skip_translated=cfg["smart_skip_translated"],
                reasoning_effort=reasoning_effort,
            )
            was_cancelled = self.stop_flag.is_set()
            prefix = "キャンセル" if was_cancelled else "完了"
            msg = (
                f"{prefix}: 書込 {result.written_files}/{result.total_files} ファイル, "
                f"キャッシュヒット {result.cache_hits}, "
                f"API {result.api_strings} 件, "
                f"失敗バッチ {result.failed_batches}"
            )
            logger.info(msg)
            self.root.after(0, lambda: self.var_progress_text.set(msg))
            self.root.after(
                0,
                lambda: messagebox.showinfo(
                    prefix,
                    f"{msg}\n\n出力先:\n{result.output_dir}",
                ),
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("翻訳処理が失敗: %s", e)
            self.root.after(0, lambda: messagebox.showerror("エラー", str(e)))
        finally:
            self.root.after(0, lambda: self.btn_start.configure(state="normal"))
            self.root.after(0, lambda: self.btn_cancel.configure(state="disabled"))

    # --------------------------------------------------------- log pumping
    def _drain_log_queue(self) -> None:
        try:
            while True:
                record = self.log_queue.get_nowait()
                try:
                    msg = record.getMessage()
                    # QueueHandler passes the raw LogRecord. Format via handler's format()
                    import logging as _lg

                    fmt = _lg.Formatter(
                        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
                    )
                    line = fmt.format(record) + "\n"

                    self.text_log.configure(state="normal")
                    self.text_log.insert("end", line, record.levelname)
                    self.text_log.see("end")
                    self.text_log.configure(state="disabled")
                except Exception:
                    pass
        except queue.Empty:
            pass
        self.root.after(100, self._drain_log_queue)

    # ================================================================= run
    def run(self) -> None:
        self.root.mainloop()


def launch() -> None:
    app = TranslatorApp()
    app.run()
