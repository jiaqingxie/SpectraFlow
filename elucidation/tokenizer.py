from __future__ import annotations

from dataclasses import dataclass


PAD = "<pad>"
BOS = "<bos>"
EOS = "<eos>"
UNK = "<unk>"
MASK = "<mask>"


@dataclass
class SmilesTokenizer:
    stoi: dict
    itos: list

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def bos_id(self) -> int:
        return self.stoi[BOS]

    @property
    def eos_id(self) -> int:
        return self.stoi[EOS]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    @property
    def mask_id(self) -> int:
        return self.stoi[MASK]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        for ch in s:
            ids.append(self.stoi.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int], stop_at_eos: bool = True) -> str:
        chars = []
        for i in ids:
            if i == self.eos_id and stop_at_eos:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            if 0 <= i < len(self.itos):
                tok = self.itos[i]
                if tok not in {PAD, BOS, EOS, UNK, MASK}:
                    chars.append(tok)
        return "".join(chars)

    def state_dict(self) -> dict:
        return {"stoi": self.stoi, "itos": self.itos}

    @classmethod
    def from_state_dict(cls, state: dict) -> "SmilesTokenizer":
        return cls(stoi=state["stoi"], itos=state["itos"])


def build_tokenizer(smiles_list: list[str]) -> SmilesTokenizer:
    charset = set()
    for s in smiles_list:
        charset.update(list(s))
    itos = [PAD, BOS, EOS, UNK, MASK] + sorted(charset)
    stoi = {tok: i for i, tok in enumerate(itos)}
    return SmilesTokenizer(stoi=stoi, itos=itos)
