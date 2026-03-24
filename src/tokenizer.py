"""Lightweight tokenizer wrapper using SentencePiece."""

from pathlib import Path
from dataclasses import dataclass


@dataclass
class TokenizerConfig:
    vocab_size: int = 32000
    model_prefix: str = "tokenizer"
    model_type: str = "bpe"
    character_coverage: float = 0.9995
    pad_id: int = 0
    unk_id: int = 1
    bos_id: int = 2
    eos_id: int = 3


class Tokenizer:
    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self.sp = None
        self.config = None

        if self.model_path and self.model_path.exists():
            self.load()

    def train(self, text_file: str | Path, config: TokenizerConfig | None = None):
        import sentencepiece as spm

        text_file = Path(text_file)
        if not text_file.exists():
            raise FileNotFoundError(f"Text file not found: {text_file}")

        self.config = config or TokenizerConfig()

        output_dir = text_file.parent.parent / "artifact"
        output_dir.mkdir(exist_ok=True)

        model_prefix = str(output_dir / self.config.model_prefix)

        print(f"Training tokenizer with vocab size: {self.config.vocab_size}")
        print(f"Input: {text_file}")
        print(f"Output: {model_prefix}.model")

        spm.SentencePieceTrainer.train(
            input=str(text_file),
            model_prefix=model_prefix,
            vocab_size=self.config.vocab_size,
            model_type=self.config.model_type,
            character_coverage=self.config.character_coverage,
            pad_id=self.config.pad_id,
            unk_id=self.config.unk_id,
            bos_id=self.config.bos_id,
            eos_id=self.config.eos_id,
            pad_piece="<pad>",
            unk_piece="<unk>",
            bos_piece="<s>",
            eos_piece="</s>",
        )

        self.model_path = Path(f"{model_prefix}.model")
        self.load()

        return self

    def load(self, model_path: str | Path | None = None):
        import sentencepiece as spm

        if model_path:
            self.model_path = Path(model_path)

        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"Tokenizer model not found: {self.model_path}")

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(self.model_path))
        print(f"Loaded tokenizer: {self.model_path}")

    @property
    def vocab_size(self) -> int:
        if not self.sp:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.get_piece_size()

    def encode(self, text: str) -> list[int]:
        if not self.sp:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.encode(text)

    def decode(self, ids: list[int]) -> str:
        if not self.sp:
            raise RuntimeError("Tokenizer not loaded")
        return self.sp.decode(ids)

    def encode_file(
        self, text_file: str | Path, save_path: str | Path | None = None
    ) -> list[int]:
        text_file = Path(text_file)
        text = text_file.read_text(encoding="utf-8", errors="replace")
        ids = self.encode(text)

        if save_path:
            import numpy as np

            save_path = Path(save_path)
            np.array(ids, dtype=np.uint16).tofile(save_path)

        return ids

    @staticmethod
    def load_bin(path: str | Path) -> "list[int]":
        import numpy as np

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Binary file not found: {path}")

        return np.fromfile(path, dtype=np.uint16).tolist()


def train_tokenizer():
    ROOT = Path(__file__).resolve().parent.parent / "data"

    config = TokenizerConfig(
        vocab_size=32000,
        model_prefix="eci_slm_tokenizer",
    )

    combined_file = ROOT / "combined_train.txt"

    print("Combining all pretraining data...")
    data_dirs = [
        ROOT / "pretrain",
        ROOT / "pretrain_expanded",
        ROOT / "pretrain_augmented",
        ROOT / "english_pretrain",
    ]

    all_texts = []
    for d in data_dirs:
        if d.exists():
            for txt_file in d.glob("*.txt"):
                all_texts.append(txt_file.read_text(encoding="utf-8", errors="replace"))

    combined_text = "\n\n".join(all_texts)
    combined_file.write_text(combined_text, encoding="utf-8")
    print(f"Combined {len(all_texts)} files, {len(combined_text):,} chars")

    tokenizer = Tokenizer()
    tokenizer.train(combined_file, config)

    print(f"\nTokenizer trained!")
    print(f"Vocab size: {tokenizer.vocab_size}")

    sample = "The Election Commission of India"
    ids = tokenizer.encode(sample)
    decoded = tokenizer.decode(ids)
    print(f"\nSample: '{sample}'")
    print(f"Encoded: {ids}")
    print(f"Decoded: '{decoded}'")


if __name__ == "__main__":
    train_tokenizer()
