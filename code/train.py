import evaluate
import argparse
import json
import os
import torch
import random
import numpy as np

from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import get_scheduler
from torch.optim import AdamW
from accelerate import Accelerator

from data import load_bioasq_dataset, preprocess_datasets, create_dataloaders
from utils import evaluate_model

import logging
from datetime import datetime



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
    seed = 1024

    torch.manual_seed(seed)            
    torch.cuda.manual_seed(seed)       
    torch.cuda.manual_seed_all(seed)

    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    config = parse_args()
    print(config)

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

    log_interval = 10

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("training")

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
    start_time = datetime.now()
    metric = evaluate.load('squad')
    # progress_bar = tqdm(range(num_training_steps), disable=disable_tqdm)

    # Evaluation at the beginning
    val_metrics = evaluate_model(
            model, validation_dataloader, validation_dataset, bioasq_dataset["validation"], metric, n_best, max_answer_length, accelerator
        )
    test_metrics = evaluate_model(
            model, test_dataloader, test_dataset, bioasq_dataset["test"], metric, n_best, max_answer_length, accelerator
        )
    # accelerator.print(f"[Log] Final Test metrics:", metrics)
    logger.info(f"First Test - Val Metrics:{val_metrics} Test Metrics: {test_metrics}")

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

            if step % log_interval == 0:
                current_lr = lr_scheduler.get_last_lr()[0]
                elapsed_time = datetime.now() - start_time
                eta = (elapsed_time / (step + 1)) * (num_training_steps - (step + 1))
    
                
                logger.info(
                    f"Epoch [{epoch+1}/{num_train_epochs}][{step+1}/{num_update_steps_per_epoch}] "
                    f"lr: {current_lr:.1e}, eta: {eta}, loss: {loss.item():.4f}"
                )

        train_losses.append(epoch_loss / len(train_dataloader))

        # Evaluation
        val_metrics = evaluate_model(
            model, validation_dataloader, validation_dataset, bioasq_dataset["validation"], metric, n_best, max_answer_length, accelerator
        )
        test_metrics = evaluate_model(
            model, test_dataloader, test_dataset, bioasq_dataset["test"], metric, n_best, max_answer_length, accelerator
        )
        validation_metrics.append(metrics)
        logger.info(f"Epoch [{epoch+1}/{num_train_epochs}][Evaluation] - Train Loss: {train_losses[-1]:.4f}, Validation Metrics: {val_metrics}, Test Metrics: {test_metrics}")


    # Evaluation on the test set
    metrics = evaluate_model(
            model, test_dataloader, test_dataset, bioasq_dataset["test"], metric, n_best, max_answer_length, accelerator
        )
    # accelerator.print(f"[Log] Final Test metrics:", metrics)
    logger.info(f"Final Test - Train Loss: {train_losses[-1]:.4f}, Test Metrics: {metrics}")
    
    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

if __name__ == "__main__":
    main()
