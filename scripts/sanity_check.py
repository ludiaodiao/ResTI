"""scripts/sanity_check.py

A tiny smoke test to confirm the repo runs end-to-end.

- Uses a small but sufficient data slice (data_use_rate=0.10)
- Trains for a few epochs only
"""

import train as T

# override a few knobs for speed
T.DATASET = "ETTh1"
T.CHANNELS = [1]
T.NUM_EPOCHS = 3
T.BATCH_SIZE = 512
T.DATA_USE_RATE = 0.10
T.REPEATS = 1
T.OUT_ROOT = "./outputs_sanity"

if __name__ == "__main__":
    T.main()
