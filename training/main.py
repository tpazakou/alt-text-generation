from config_loader import load_config
from train import *
import torch

# model
from transformers import LlavaForConditionalGeneration
from transformers import AutoProcessor


def main():
    config = load_config()

    # ----------------- model set up --------------------------------------

    base_model_name = config.base_model_name
    processor = AutoProcessor.from_pretrained(base_model_name, use_fast=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model.gradient_checkpointing_enable()

    model_wrapper = LoRALLaVAWrapper(
        model=model,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout
    )
 

    # ----------------- data -----------------------------------------------
    data_processor = DataProcessor(processor=processor, config=config)
    training_set = data_processor.get_dataloader(split="train")
    valid_set = data_processor.get_dataloader(split="val")

    # ------------------ trainer -------------------------------------------
    dpo_loss = DPOLoss(beta=config.beta)

    trainer = Trainer(
        model=model_wrapper.model,
        train_dataloader=training_set,
        val_dataloader=valid_set,
        dpo_loss=dpo_loss,
        lr=float(config.learning_rate),
        weight_decay=config.weight_decay,
        device=config.device,
        output_dir=config.output_dir,
        max_grad_norm=config.max_grad_norm,
        save_best_only=True,
        early_stopping_patience=config.patience,
        accumulation_steps=config.accumulation_steps
    )

    trainer.train(num_epochs=config.num_epochs)


if __name__ == "__main__":
    main()
