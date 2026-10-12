
import sys
import json
import csv
import time
import random
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from contextlib import contextmanager
from datetime import datetime

import torch
import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may reduce performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@contextmanager
def timer(name: str = "Operation", verbose: bool = True, use_cuda_sync: bool = True):
    timing_info = {"name": name, "start": None, "end": None, "elapsed": None}
    
    if use_cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
        
    timing_info["start"] = time.perf_counter()
    
    try:
        yield timing_info
    finally:
        if use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        timing_info["end"] = time.perf_counter()
        timing_info["elapsed"] = timing_info["end"] - timing_info["start"]
        
        if verbose:
            print(f"{name}: {timing_info['elapsed']:.3f}s")


class Timer:
    def __init__(self, use_cuda_sync: bool = True):
        self.use_cuda_sync = use_cuda_sync
        self._start_time = None
        self._elapsed = 0.0
        
    def start(self) -> "Timer":
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start_time = time.perf_counter()
        return self
        
    def stop(self) -> float:
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
            
        if self.use_cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        self._elapsed = time.perf_counter() - self._start_time
        self._start_time = None
        return self._elapsed
        
    @property
    def elapsed(self) -> float:
        return self._elapsed


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(data: List[Dict], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def load_jsonl(path: Union[str, Path]) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_csv(
    data: List[Dict],
    path: Union[str, Path],
    fieldnames: Optional[List[str]] = None,
) -> None:
    if not data:
        return
        
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if fieldnames is None:
        # Collect all unique keys preserving order
        fieldnames = []
        seen = set()
        for row in data:
            for key in row.keys():
                if key not in seen:
                    fieldnames.append(key)
                    seen.add(key)
                    
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(data)


def load_csv(path: Union[str, Path]) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path))
        
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )
    
    return logging.getLogger()


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Union[str, Path]) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Union[str, Path]) -> Dict:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: Union[str, Path]) -> None:
    import yaml

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


def format_number(n: float, decimals: int = 2) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    elif n >= 1_000:
        return f"{n/1_000:.{decimals}f}K"
    else:
        return f"{n:.{decimals}f}"


def truncate_string(s: str, max_len: int = 100, suffix: str = "...") -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len - len(suffix)] + suffix
