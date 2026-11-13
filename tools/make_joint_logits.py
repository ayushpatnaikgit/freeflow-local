"""Rebuild ane/JointDecisionv3Logits.mlmodel from Oruk AI's JointDecisionv3.mlpackage (needs coremltools).

The stock joint model computes scores for all 8,193 word-pieces, then returns only the argmax (token_id), its
probability and the duration. Vocabulary boosting needs the scores, so this adds one output, `token_logits`, as an
identity of the existing `token_logits_cast_fp16` variable. No weights change: setup-ane.sh copies Oruk AI's package
and swaps in this graph description only. The check at the end confirms the stock outputs are unchanged.

Usage: python tools/make_joint_logits.py <JointDecisionv3.mlpackage> <out .mlmodel>
"""

import shutil
import sys
import tempfile
from pathlib import Path

import coremltools as ct
import numpy as np
from coremltools.proto import FeatureTypes_pb2 as ft


def main() -> None:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    spec = ct.models.MLModel(str(src), skip_model_load=True).get_spec()
    o = spec.description.output.add()
    o.name = "token_logits"
    o.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT16
    o.type.multiArrayType.shape.extend([1, 1, 1, 8193])
    block = spec.mlProgram.functions["main"].block_specializations[spec.mlProgram.functions["main"].opset]
    op = block.operations.add()
    op.type = "identity"
    op.inputs["x"].arguments.add().name = "token_logits_cast_fp16"
    var = op.outputs.add()
    var.name = "token_logits"
    var.type.CopyFrom(next(v for o2 in block.operations for v in o2.outputs if v.name == "token_logits_cast_fp16").type)
    block.outputs.append("token_logits")

    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "JointDecisionv3Logits.mlpackage"
        ct.models.MLModel(spec, weights_dir=str(src / "Data/com.apple.CoreML/weights"), skip_model_load=True).save(str(pkg))
        stock, new = ct.models.MLModel(str(src)), ct.models.MLModel(str(pkg))
        rng = np.random.default_rng(0)
        for _ in range(50):
            x = {"encoder_step": rng.standard_normal((1, 1024, 1)).astype(np.float32),
                 "decoder_step": rng.standard_normal((1, 640, 1)).astype(np.float32)}
            a, b = stock.predict(x), new.predict(x)
            assert a["token_id"].item() == b["token_id"].item() == int(np.argmax(b["token_logits"]))
            assert a["duration"].item() == b["duration"].item()
        shutil.copy(pkg / "Data/com.apple.CoreML/model.mlmodel", out)
    print(f"wrote {out}; stock outputs unchanged on 50 random inputs")


if __name__ == "__main__":
    main()
