import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration


class SummaryModel(pl.LightningModule):
    def __init__(self, model_name="t5-base", total_step=None):
        super(SummaryModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, return_dict=True)
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
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["summary_ids"]
        decoder_attention_mask = batch["summary_mask"]

        loss, output = self(input_ids, attention_mask, labels, decoder_attention_mask)
        self.log("val_loss", loss.item())

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        loss, output = self(input_ids=input_ids, attention_mask=attention_mask)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0,
            num_training_steps=self.total_step)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}