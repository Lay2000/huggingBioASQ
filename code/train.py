import evaluate
import argparse
import json
import os

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import get_scheduler
from torch.optim import AdamW
from accelerate import Accelerator

from data import load_bioasq_dataset, preprocess_datasets, create_dataloaders
from utils import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a model on the BioASQ dataset.")
    parser.add_argument("--config", type=str, default="config.json", help="Path to the configuration file.")
    args = parser.parse_args()
    def read_config_file(file_path):
        with open(file_path, 'r') as file:
            config = json.load(file)
        return config

    # Read the configuration file
    config = read_config_file(args.config)

    return config


def main():
    config = parse_args()

    # Configuration
    model_checkpoint = config["model"]["model_checkpoint"]

    task_type = config["data"]["task_type"]
    max_length = config["data"]["max_length"]
    stride = config["data"]["stride"]

    batch_size = config["hyperparameters"]["batch_size"]
    train_epochs = config["hyperparameters"]["train_epochs"]
    lr = config["hyperparameters"]["lr"]
    optimizer_type = config["hyperparameters"]["optimizer"]
    scheduler_type = config["hyperparameters"]["scheduler"]
    num_warmup_steps = config["hyperparameters"]["num_warmup_steps"]

    n_best = config["others"]["n_best"]
    max_answer_length = config["others"]["max_answer_length"]
    output_dir = config["others"]["output_dir"]

    disable_tqdm = os.environ.get("DISABLE_TQDM", "False") == "True"

    # Load and preprocess datasets
    bioasq_dataset = load_bioasq_dataset(task_type)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_dataset, validation_dataset, test_dataset = preprocess_datasets(tokenizer, bioasq_dataset, max_length, stride)
    train_dataloader, validation_dataloader, test_dataloader = create_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size)

    # Load model, optimizer, scheduler, and accelerator
    model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
    # Set the optimizer based on the configuration
    if optimizer_type == "AdamW":
        optimizer = AdamW(model.parameters(), lr=lr)
    # Add more optimizer types here if needed

    accelerator = Accelerator()
    model, optimizer, train_dataloader, validation_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader, test_dataloader
    )

    num_train_epochs = train_epochs
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    # Set the scheduler based on the configuration
    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Training and evaluation loop
    metric = evaluate.load('squad')
    progress_bar = tqdm(range(num_training_steps), disable=disable_tqdm)

    train_losses = []
    validation_metrics = []

    for epoch in range(num_train_epochs):
        # Training
        model.train()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            outputs = model(**batch)
            loss = outputs.loss
            epoch_loss += loss.item()
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            if step % 10 == 0:
                progress_bar.set_description(f"Epoch {epoch+1}/{num_train_epochs}")
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                progress_bar.update(10)

        train_losses.append(epoch_loss / len(train_dataloader))

        # Evaluation
        metrics = evaluate_model(
            model, validation_dataloader, validation_dataset, bioasq_dataset["validation"], metric, n_best, max_answer_length, accelerator
        )
        validation_metrics.append(metrics)
        accelerator.print(f"[Log] Epoch {epoch} - Train Loss: {train_losses[-1]:.4f}, Validation Metrics: {metrics}")


    # Evaluation on the test set
    metrics = evaluate_model(
            model, test_dataloader, test_dataset, bioasq_dataset["test"], metric, n_best, max_answer_length, accelerator
        )
    accelerator.print(f"[Log] Final Test metrics:", metrics)

if __name__ == "__main__":
    main()
