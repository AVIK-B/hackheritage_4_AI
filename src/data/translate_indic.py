"""Batch en->indic translation via AI4Bharat IndicTrans2. Runs on Kaggle GPU.
If the toolkit is unavailable, returns inputs unchanged so pipelines never block."""

MODEL = "ai4bharat/indictrans2-en-indic-1B"


class IndicTranslator:
    def __init__(self, tgt_lang: str = "hin_Deva", batch_size: int = 32):
        self.ok = False
        try:
            import torch
            from IndicTransToolkit import IndicProcessor
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            self.torch = torch
            self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).eval().cuda()
            self.ip = IndicProcessor()
            self.tgt, self.bs = tgt_lang, batch_size
            self.ok = True
        except Exception as e:
            print(f"[translator] disabled, passthrough mode: {e}")

    def __call__(self, texts: list[str]) -> list[str]:
        if not self.ok:
            return texts
        out = []
        for i in range(0, len(texts), self.bs):
            batch = self.ip.preprocess_batch(texts[i : i + self.bs], src_lang="eng_Latn", tgt_lang=self.tgt)
            enc = self.tok(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
            with self.torch.no_grad():
                gen = self.model.generate(**enc, num_beams=4, max_new_tokens=256)
            out += self.ip.postprocess_batch(gen, lang=self.tgt)
        return out


def translate_file(src: str, dst: str, field: str, tgt_lang: str = "hin_Deva") -> None:
    import json
    import pathlib

    rows = [json.loads(line) for line in pathlib.Path(src).read_text().splitlines() if line.strip()]

    tr = IndicTranslator(tgt_lang)
    texts = [r[field] for r in rows]
    translated = tr(texts) if tr.ok else texts
    for r, t in zip(rows, translated, strict=True):
        r[field] = t
        if tr.ok:
            r["lang"] = tgt_lang[:2]
    pathlib.Path(dst).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows))
    print(f"{'translated' if tr.ok else 'passthrough'} {len(rows)} rows -> {dst}")


if __name__ == "__main__":
    import sys

    translate_file(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "target")
