# imports

# DataProcessor
from torch.utils.data import Dataset
import torch.nn as nn
from PIL import Image
import json
import torch
import zipfile
import os
import gdown
import shutil

# DPO Loss
import torch.nn.functional as fn

# Trainer
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from peft import get_peft_model, LoraConfig, TaskType

# status updates
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)


# ------------------- Wrapper -----------------------------

class LoRALLaVAWrapper(nn.Module):
    """
    Loads the model and handles the freezing and modified components.

      - Wraps the LLaVA model for finetuning
      - LoRA adapters applied only to attention layers of the LLM
      - Full finetuning of the vision-to-text projection layer (mm_projector)
      - Freezes the rest of the parameters for efficient tuning
      - Handles forward pass
      - Provides a generate method for easy sequence generation / inference.
    """
    def __init__(self, model, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()

        # Loading the base model
        self.model = model
        total_params = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Initial model loaded successfully with {total_params:,} parameters.")

        # freezing all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # lora configuration
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"],  # targeting only the attention layers (query and value matrices)
        )

        # apply the lora adapters
        logging.info("Applying LoRA adapters.")
        self.model = get_peft_model(self.model, lora_config)

        # unfreeze lora matrices and projection layer
        self._unfreeze_lora_and_projection()

    def _unfreeze_lora_and_projection(self):
        """
          Unfreezes the lora and projection layers
        """
        # unfreeze all lora adapters of the language model
        for name, param in self.model.named_parameters():
            if "lora_" in name and "vision_tower" not in name:
                param.requires_grad = True
                logging.debug(f"Unfroze LoRA parameter: {name}")

        # unfreeze projection layer
        for name, module in self.model.named_modules():
            if "projector" in name:
                for param in module.parameters():
                    param.requires_grad = True
                    logging.debug(f"Unfroze projector parameter: {name}")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"After setup: {trainable_params:,} of the total {total_params:,} parameters are trainable.")

    def forward(self, images, input_ids, attention_mask):
        """
            Forward pass. DPO requires 2 forward passes (one per response), so it will be called twice in the Trainer.
            There is no need for labels, loss is manually calculated in the DPO class

            Args:
              images (torch.Tensor): image tensor (batch_size, 3, H, W) -> (batch_size, 3(RGB), 224, 224)
              input_ids (torch.Tensor): text input IDs
              attention_mask (torch_tensor): attention mask for text tokens

            Returns:
              model outputs: logits, loss etc.
          """
        return self.model(
             input_ids=input_ids,
             attention_mask=attention_mask,
             pixel_values=images,
         )

    def generate(self, images, input_ids=None, attention_mask=None, **generate_kwargs):
        """
          Generates sequences using the wrapped LLaVA model.
          Input_ids and attention_mask corresponds only to the instruction and the context if applicable.
        """
        return self.model.generate(
              pixel_values=images,
              input_ids=input_ids,
              attention_mask=attention_mask,
              **generate_kwargs
          )

# ------------------- Data loading -----------------------------


class DataProcessor:
    """
      Downloads images, loads the dataset from the json file.
      Encodes the input that has previously been formatted in a DPO appropriate format using DPOPairwiseDataset.

      Args:
        processor: processor for both the image and the text components
        config: configuration setup file

      Returns:
        - Pytorch DataLoaders for the requested split
    """

    def __init__(self, processor, config):
        self.config = config
        self.processor = processor
        self.image_zip_path = config.image_zip_path
        self.dataset = self.load_dataset(config.dataset_path)
        self._download_images(config.image_zip_url, config.image_zip_path, config.img_unzipped_path, config.images_path)

    def _download_images(self, img_url, img_local_path, img_unzipped_path, images_path):
        """
            It downloads the Concadia zip from google drive if it doesn't already exist locally.
            It unzips the folder and extracts the images into a nested resized file.
            It adds the resized file to the images folder already containing the ad2at_md images.
            Final image folder organisation:
            images/
            ├── resized/
            │   └── 1.jpg
            └── 1004_Juno/
                └── 1004_Juno_00.00.32.849-00.00.35.458/
                    └── 0009.jpg
        """

        # downloading the concadia zip
        if not os.path.exists(img_local_path):
            logging.info(f"Downloading image zip for Concadia images to {img_local_path}...")
            gdown.download(img_url, img_local_path, quiet=False, fuzzy=True)
        else:
            logging.info(f"Image zip for Concadia images already exists at {img_local_path}, skipping download.")

        # creating the folder path for the extracted images if it doesn't exist already
        os.makedirs(img_unzipped_path, exist_ok=True)
        resized_path = "../input/images/resized/"

        # extracting the concadia images in the images folder
        if not os.path.exists(resized_path) or len(os.listdir(resized_path)) == 0:
            with zipfile.ZipFile(img_local_path, "r") as zip_ref:
                files_to_extract = ["resized/" + row["img_name"] for row in self.dataset if row["dataset"] == "concadia"]
                if files_to_extract:
                    for file in files_to_extract:
                        zip_ref.extract(file, img_unzipped_path)
                else:
                    logging.info("The archive is empty. Please load a valid archive.")
            images = os.listdir(os.path.join(img_unzipped_path, "resized"))
            logging.debug(f"A total of {len(images)} Concadia images were extracted.")

            # adding the resized folder into the ad2at_images folder
            shutil.move(os.path.join(img_unzipped_path, "resized"), images_path)

            # final count of extracted images
            count = 0
            for root, dirs, files in os.walk(images_path):
                count += sum(file.lower().endswith('.jpg') for file in files)
            logging.debug(f"The final image count between the two datasets is {count}.")
        else:
            logging.info(f"Images already extracted, skipping extraction.")

    @staticmethod
    def load_dataset(json_path):
        """
        Reads the JSONL file

        Args:
          - json_path: the path to the jsonl file

        Returns:
          - list of dicts: each dict corresponds to a line from the jsonl file
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
            logging.info(f"Loaded {len(data)} examples from the jsonl file.")
            return data

    def get_dataloader(self, split="train"):
        """
        Creates a Pytorch DataLoader for the given data split.

        Args:
            - split: the split for which the dataloader is created

        Returns:
            - the dataloader for the given split
        """

        # filtering per split
        filtered_data = [item for item in self.dataset if item.get("split") == split][:10000]
        logging.info(f"The {split} split contains {len(filtered_data)} examples.")

        # properly initialize the dataset for DPO training
        dataset = DPOPairwiseDataset(
            config=self.config,
            data=filtered_data,
            processor=self.processor,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),  # shuffle only when training
            collate_fn=dataset.collate_fn
        )


class DPOPairwiseDataset(Dataset):
    """

    Dataset class for DPO (Direct Preference Optimization) tuning.
    Formats the dataset in DPO appropriate format.

    """

    def __init__(self, config, data, processor):

        self.processor = processor
        self.config = config
        self.image_folder = config.images_path
        self.data = data
        self.device = config.device
        self._logged_once = False

    def __len__(self):
        """ Returns the number of examples in the dataset """
        return len(self.data)

    def __getitem__(self, index):
        """Loads and formats a single sample"""
        example = self.data[index]

        # --------------------- extracting text from json --------------------------

        # Extract prompt and responses from the sample
        image_name = example["img_name"]
        prompt = example["prompt"]
        # if context is not provided, falls back to empty string
        context = example.get("context", "")
        chosen = example["chosen"]
        rejected = example["rejected"]

        # --------------------- image ---------------------------------------------

        image_file = None  # default
        if example["dataset"] == "concadia":
            image_file = "resized/" + image_name
        elif example["dataset"] == "ad2at_md":
            folder = example["img_name"][:example["img_name"].rfind("_")]
            image_file = folder + "/" + example['img_name']

        image_path = os.path.join(self.image_folder, image_file)
        if not os.path.exists(image_path):
            logging.warning(f"Image not found: {image_path}. Skipping sample.")
            return None

        image = Image.open(image_path).convert("RGB")

        # --------------------- creating the full prompt ---------------------------

        try:
            full_prompt_chosen = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",
                         "text": prompt + ". The image appeared in the following context: " + context.strip()},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": chosen.strip()},
                    ],
                },
            ]

            full_prompt_rejected = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text",
                         "text": prompt + ". The image appeared in the following context: " + context.strip()},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": rejected.strip()},
                    ],
                },
            ]
        except Exception as e:
            logging.warning(f"Error processing {image_path}: {e}. Skipping sample.")
            return None

        return full_prompt_chosen, full_prompt_rejected

    def collate_fn(self, batch):

        """
        Batch of samples where each sample is a list (length=1) with a dict
        containing 'role' and 'content' list, including raw PIL images.

        We extract images and texts, then process batches with the processor.
        """

        # filtering out missing examples if applicable
        batch = [(chosen, rejected) for chosen, rejected in batch if chosen and rejected]

        # debugging
        example_chosen, example_rejected = batch[0]
        logging.debug(f"Chosen example before encoding: {example_chosen}")
        logging.debug(f"Rejected example before encoding: {example_rejected}")

        # batch: List of (full_prompt_chosen, full_prompt_rejected) pairs from __getitem__
        full_prompt_chosen_batch, full_prompt_rejected_batch = zip(*batch)

        # Process chosen batch
        chosen_inputs = self.processor.apply_chat_template(
            list(full_prompt_chosen_batch),
            tokenize=True,
            return_dict=True,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)

        # Process rejected batch
        rejected_inputs = self.processor.apply_chat_template(
            list(full_prompt_rejected_batch),
            tokenize=True,
            return_dict=True,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        ).to(self.device)

        # verifying shapes after encoding
        if not self._logged_once:
            # chosen batch
            logging.debug(f"Batch shape for the chosen_inputs after encoding: {list(chosen_inputs.keys())}")
            logging.debug(f"In the batch: input_ids : {chosen_inputs['input_ids'].shape}")
            logging.debug(f"attention_mask : {chosen_inputs['attention_mask'].shape}")
            logging.debug(f"pixel_values : {chosen_inputs['pixel_values'].shape}")

            # rejected batch
            logging.debug(f"Batch shape for the rejected_inputs after encoding: {list(rejected_inputs.keys())}")
            logging.debug(f"In the batch: input_ids : {rejected_inputs['input_ids'].shape}")
            logging.debug(f"attention_mask : {rejected_inputs['attention_mask'].shape}")
            logging.debug(f"pixel_values : {rejected_inputs['pixel_values'].shape}")

            self._logged_once = True
        
	# dictionnary keys: ['input_ids', 'attention_mask', 'pixel_values']
        return chosen_inputs, rejected_inputs

# ------------------------- DPO Loss -------------------------------


class DPOLoss(nn.Module):
    """
        Calculates DPO loss
    """
    def __init__(self, beta=0.1):

        """
          Initializes the DPO loss calculator and saves beta for use in the loss function.

          Args:
              beta (β in the loss function): Scaling factor that controls the sharpness of the preference margin.
              The higher the β, the more the model is punished and the more it becomes confident
              in choosing the chosen sample.
        """
        super().__init__()
        self.beta = beta

    @staticmethod
    def get_log_probs(logits, labels, attention_mask):
        """
          Computes log-probabilities for a given target sequence.

          Args:
              logits (Tensor): the raw logits (scores) output by the model (batch_size, seq_len, vocab_size),
                               i.e. how confident the model is for every word it generates
              labels (Tensor): Target token ids (batch_size, seq_len)
                               i.e. the actual words the model should have predicted
              attention_mask (Tensor): Attention mask (batch_size, seq_len) indicating which tokens to include.
                                       1: the token should be included in the loss calculation, 0: no

          Returns:
              Tensor: log-prob sums for each sequence of the batch
        """

        # Apply log-softmax to transform the logits into logarithms of probabilities
        log_probs = fn.log_softmax(logits, dim=-1)

        # Extract the log-probability assigned to each token for all tokens of the target sequence
        # i.e. how confident the model was when it predicted the right token
        label_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        # Now shape is (batch_size, sequence_length - 1)
        # logging.info(f"label_log_probs shape: {label_log_probs.shape}")

        # Zero out log-probs where attention_mask == 0 (padding tokens)
        masked_log_probs = label_log_probs * attention_mask

        # Sum the log-probs over the correctly predicted(non-padded) tokens for each sequence
        return masked_log_probs.sum(dim=-1)  # (batch_size,)

    def forward(self,
                chosen_logits, chosen_labels, chosen_attention_mask,
                rejected_logits, rejected_labels, rejected_attention_mask):

        """
            Calculates the DPO loss based on the model's predicted logits for the chosen and rejected sequences

            Returns:
                Tensor: Scalar DPO loss to be minimized
        """

        # Get total log-likelihood for chosen and rejected sequences
        logp_chosen = self.get_log_probs(chosen_logits, chosen_labels, chosen_attention_mask)
        logging.debug(f"logp_chosen: {logp_chosen}")
        logp_rejected = self.get_log_probs(rejected_logits, rejected_labels, rejected_attention_mask)
        logging.debug(f"logp_rejected: {logp_rejected}")
        
        len_chosen = chosen_attention_mask.sum(dim=1).float().clamp(min=1)
        len_rejected = rejected_attention_mask.sum(dim=1).float().clamp(min=1)
        logp_chosen = logp_chosen / len_chosen
        logging.debug(f"logp_chosen: {logp_chosen}")
        logp_rejected = logp_rejected / len_rejected
        logging.debug(f"logp_rejected : {logp_rejected}")

        # Calculate the margin scaled by beta
        diff = (logp_rejected - logp_chosen) * self.beta
        logging.debug(f"Difference: {diff}")

        # Compute the loss with the softplus function applied to the difference
        loss = torch.nn.functional.softplus(diff).mean()  # softplus(x) = log(1 + exp(x))
        logging.debug(f"Loss: {loss}")

        # This is what gets back propagated through the model
        return loss

# ------------------------------- Trainer ------------------------------------------------


class Trainer:
    def __init__(
      self,
      model: nn.Module,  # LoRA-wrapped LLaVA model
      train_dataloader: DataLoader,  # DataLoader for training data
      val_dataloader: DataLoader,  # DataLoader for validation data
      dpo_loss,  # Instance of the DPOLoss class
      lr: float = 1e-4,  # Learning rate for the optimizer
      weight_decay: float = 0.0,  # Weight decay for the optimizer
      device: str = "cuda",
      output_dir: str = "./checkpoints",
      max_grad_norm: float = 1.0,
      save_best_only: bool = True,
      early_stopping_patience: int = None,
      accumulation_steps: int = 4
    ):

        """
            Initializes the Trainer.
        """

        # Move model to the target device
        self.model = model.to(device)

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.dpo_loss = dpo_loss
        self.device = device
        self.save_best_only = save_best_only
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.early_stopping_patience = early_stopping_patience
        self.no_improve_count = 0
        self.accumulation_steps=accumulation_steps

        # Ensure checkpoint directory exists
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # set the optimizer. It only updates parameters with requires_grad=True (LoRA + projector)
        self.optimizer = AdamW(
                  filter(lambda p: p.requires_grad, model.parameters()),
                  lr=lr,
                  weight_decay=weight_decay,
              )

        self.max_grad_norm = max_grad_norm
        logging.info(f"Trainer initialized on device {device}")

    def train_epoch(self):

        """
            Runs one training epoch. Returns the average loss for that epoch.
        """

        self.model.train()  # setting model to training mode
        losses = []  # to store losses for all batches
        self.optimizer.zero_grad()

        for step, batch in enumerate(tqdm(self.train_dataloader, desc="Training")):
            chosen_input, rejected_input = batch

            # Move batch tensors to the device
            images = chosen_input["pixel_values"].to(self.device)
            chosen_input_ids = chosen_input["input_ids"].to(self.device)
            chosen_attention_mask = chosen_input["attention_mask"].to(self.device)
            rejected_input_ids = rejected_input["input_ids"].to(self.device)
            rejected_attention_mask = rejected_input["attention_mask"].to(self.device)

            # forward pass for the batch
            chosen_outputs = self.model(
                input_ids=chosen_input_ids,
                attention_mask=chosen_attention_mask,
                pixel_values=images
            )
            chosen_logits = chosen_outputs.logits
            del chosen_outputs

            # forward pass 
            rejected_outputs = self.model(
                input_ids=rejected_input_ids,
                attention_mask=rejected_attention_mask,
                pixel_values=images
            )
            rejected_logits = rejected_outputs.logits
            del rejected_outputs, images

            # Calculate DPO loss
            loss = self.dpo_loss(
                      chosen_logits, chosen_input_ids, chosen_attention_mask,
                      rejected_logits, rejected_input_ids, rejected_attention_mask
                  )

            print("Raw batch loss:", loss.item())
            # Backpropagation
            loss = loss / self.accumulation_steps
            loss.backward()
            del rejected_logits, chosen_logits, chosen_attention_mask, rejected_attention_mask, chosen_input_ids, rejected_input_ids


            if (step + 1) % self.accumulation_steps == 0:
                # Gradient clipping to avoid exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Update model parameters
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Store the loss of the batch
                losses.append(loss.item() * self.accumulation_steps)
                del loss

        if (step + 1) % self.accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            # Store the loss of the batch
            losses.append(loss.item() * self.accumulation_steps)
            del loss

        torch.cuda.empty_cache()
        # Calculate average loss over the epoch
        avg_loss = sum(losses) / len(losses)
        return avg_loss

    def eval_epoch(self):

        """
            Runs one evaluation epoch on the validation set. Returns the average validation loss for that epoch.
            It evaluates whether the model actually prefers the preferred answer.
        """
        self.model.eval()  # set model to evaluation mode
        losses = []

        # Disable gradient computation for validation
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                chosen_input, rejected_input = batch
                images = chosen_input["pixel_values"].to(self.device)
                chosen_input_ids = chosen_input["input_ids"].to(self.device)
                chosen_attention_mask = chosen_input["attention_mask"].to(self.device)
                rejected_input_ids = rejected_input["input_ids"].to(self.device)
                rejected_attention_mask = rejected_input["attention_mask"].to(self.device)

                chosen_outputs = self.model(
                    input_ids=chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    pixel_values=images
                )
                chosen_logits = chosen_outputs.logits
                del chosen_outputs

                rejected_outputs = self.model(
                    input_ids=rejected_input_ids,
                    attention_mask=rejected_attention_mask,
                    pixel_values=images
                )
                rejected_logits = rejected_outputs.logits
                del rejected_outputs


                # Compute DPO loss
                loss = self.dpo_loss(
                            chosen_logits, chosen_input_ids, chosen_attention_mask,
                            rejected_logits, rejected_input_ids, rejected_attention_mask
                        )
                print("Raw batch loss:", loss.item())
                losses.append(loss.item())

                del chosen_logits, rejected_logits
                del chosen_input_ids, chosen_attention_mask, rejected_input_ids, rejected_attention_mask

        avg_loss = sum(losses) / len(losses)
        torch.cuda.empty_cache()
        return avg_loss

    def save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.output_dir, f"best_model.pt")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": epoch,
            }, checkpoint_path
        )

    def train(self, num_epochs):
        """
            Runs full training.
        """

        for epoch in range(1, num_epochs + 1):
            logging.info(f"Starting epoch {epoch}/{num_epochs}")
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            logging.info(f"Epoch {epoch} Train Loss: {train_loss:.4f}")

            val_loss = self.eval_epoch()
            self.val_losses.append(val_loss)
            logging.info(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

            # Run evaluation and save checkpoint
            if self.save_best_only and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch)
                logging.info(f"New best checkpoint saved at epoch {epoch} with val loss {val_loss:.4f}")
                self.no_improve_count = 0  # reset since val loss improved
            else:
                self.no_improve_count += 1

            if self.early_stopping_patience is not None and self.no_improve_count >= self.early_stopping_patience:
                logging.info(f"Early stopping triggered after {self.no_improve_count} epochs with no improvement.")
                break

        loss_df = pd.DataFrame({
            "train_loss": self.train_losses,
            "val_loss": self.val_losses
        })
        loss_df.to_csv(os.path.join(self.output_dir, "losses.csv"), index=False)

        plt.figure()
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss per Epoch")
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "loss_over_epochs.png"))
        plt.close()
