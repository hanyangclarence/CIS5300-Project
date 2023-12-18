from typing import Union, IO, Optional, Any
from typing_extensions import Self

import pytorch_lightning as pl
from lightning_fabric.utilities.types import _PATH, _MAP_LOCATION_TYPE
from transformers import get_linear_schedule_with_warmup, AdamW


class SummaryModel(pl.LightningModule):
    def __init__(self, model, total_step):
        super(SummaryModel, self).__init__()
        self.model = model
        self.total_step = total_step

    def forward(self, input_ids, attention_mask, labels=None, decoder_attention_mask=None):
        outputs = self.model(input_ids=input_ids,
                             attention_mask=attention_mask,
                             labels=labels,
                             decoder_attention_mask=decoder_attention_mask)
        return outputs.loss, outputs.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["summary_ids"]
        decoder_attention_mask = batch["summary_mask"]

        loss, output = self(input_ids, attention_mask, labels, decoder_attention_mask)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["summary_ids"]
        decoder_attention_mask = batch["summary_mask"]

        loss, output = self(input_ids, attention_mask, labels, decoder_attention_mask)
        self.log("val_loss", loss)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss, output = self(input_ids=input_ids, attention_mask=attention_mask)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=0.0001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=self.total_step)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}