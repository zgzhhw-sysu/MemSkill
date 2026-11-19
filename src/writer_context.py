"""Partition oversized writer requests within one Controller decision and output budget."""
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import re

from transformers import AutoTokenizer


class WriterContext:
    def __init__(self, model_directory):
        directory = Path(model_directory)
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        self.limit = json.loads((directory / "config.json").read_text())["max_position_embeddings"]
        files = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
        self.signature = {"algorithm": "source-ranges-v1", "context_tokens": self.limit,
                          "tokenizer_files": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                              for name in files if (directory / name).is_file()}}

    def prepare(self, source, build_prompt, max_output):
        """Return original character ranges, prompts, exact input lengths, and per-part output caps."""
        if not 0 < max_output < self.limit:
            raise ValueError("Writer output budget must be positive and smaller than its context")

        def measure(text):
            prompt = build_prompt(text)
            messages = [{"role": "user", "content": prompt}]
            tokens = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            return prompt, len(tokens)

        whole, tokens = measure(source)
        budget = self.limit - max_output
        if tokens <= budget:
            return [(0, len(source), whole, tokens, max_output)]

        header, sep, _ = source.partition("\n\n")
        header = header + sep if sep and header.startswith("Session Date: ") else ""
        roles = [(m.start(1), m.group(1)) for m in re.finditer(r"(?:^|\n\n)(user|assistant): ", source)]
        starts = [start for start, _ in roles]
        parts = []
        pos = len(header)
        while pos < len(source):
            prefix = header
            index = bisect_right(starts, pos) - 1
            if index >= 0 and starts[index] < pos:
                prefix += roles[index][1] + ": "
            end = len(source)
            prompt, tokens = measure(prefix + source[pos:end])
            if tokens > budget:
                lo, hi = pos, end
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    candidate, length = measure(prefix + source[pos:mid])
                    if length <= budget:
                        lo, prompt, tokens = mid, candidate, length
                    else:
                        hi = mid
                if pos == lo:
                    raise ValueError("Writer context cannot fit even one source character with these skills and memories")
                end = lo
                # Break at a message, newline or word boundary when it is near the end.
                near = pos + (end - pos) // 2
                rank = bisect_right(starts, end) - 1
                boundary = starts[rank] if rank >= 0 and starts[rank] > near else 0
                if not boundary:
                    boundary = source.rfind("\n", near, end) + 1 or source.rfind(" ", near, end) + 1
                if pos < boundary < end:
                    candidate, length = measure(prefix + source[pos:boundary])
                    if length <= budget:
                        end, prompt, tokens = boundary, candidate, length
            parts.append((pos if parts else 0, end, prompt, tokens))
            pos = end
        if not parts or len(parts) > max_output:
            raise ValueError("The writer output budget cannot cover every source fragment")
        per_part, remainder = divmod(max_output, len(parts))
        return [(*part, per_part + int(i < remainder)) for i, part in enumerate(parts)]
