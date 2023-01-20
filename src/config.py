import pandas as pd
from transformers import DebertaV2TokenizerFast

df = pd.read_csv("../input/train_3.csv")
MODEL_SAVE_PATH = 'model.bin'
MODEL_PATH = '../input/Deberta-v3-large/'
tokenizer = DebertaV2TokenizerFast.from_pretrained(MODEL_PATH)
TRAIN_BATCH_SIZE = 2
VAL_BATCH_SIZE = 2
TEST_BATCH_SIZE = 2
seed = 42
epochs = 10
n_splits = 5
learning_rate = 7.5e-6
fold = 0

