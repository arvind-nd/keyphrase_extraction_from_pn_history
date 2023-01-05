import .config
import pandas as pd
import train

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmpup




for fold in range(config.n_splits):
    train = config.df[config.df.kfold != fold].reset_index(drop=True)
    valid = config.df[config.df.kfold == fold].reset_index(drop=True)


    train_data = NBMEDataset(train, config.tokenizer)
    valid_data = NBMEDataset(valid, config.tokenizer)

    train_dataloder = DataLoader(
        train_dataloader,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=config.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataloader,
        batch_size=config.VAL_BATCH_SIZE,
        num_workers=config.num_workers
    )

    model = Deberta_base()
    model.to(config.device)

    param_optimize = list(model.named_parameters())
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_parameters = [
        {
            "params":[
                p for n, p in param_optimize if not any(nd in n for nd in no_decay)
                ],
            "weight_decay":0.01
            }
    ]

    optimizer = AdamW(optimizer_parameters, lr=config.learning_rate)
    num_training_steps = int(len(train)) / config.TRAIN_BATCH_SIZE * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_step=0,
        num_training_steps=num_training_steps
        )

    best_score = 0
    for epoch in range(config.epochs):
        print(f"fold/epoch = {fold}/{epoch}")
        tr_loss = train_fn(train_dataloader, model, optimizer, scheduler, config.device)
        gc.collect()
        if config.device=="cuda":
            torch.cuda.empty_cache()

        micro_average_f1_score = eval_fn(valid_dataloader, model, config.device)
        gc.collect()
        if config.device=="cuda":
            torch.cuda.empty_cache()

        print(f"valid_score = {micro_average_f1_score}")
        if best_score<micro_average_f1_score:
            best_score = micro_average_f1_score
            torch.save(model.state_dict(), config.device)
    print(f"best_score = {best_score}")
    break
