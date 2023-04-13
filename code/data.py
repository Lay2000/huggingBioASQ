from datasets import Dataset, DatasetDict, load_dataset
from functools import partial
from torch.utils.data import DataLoader
from transformers import default_data_collator

def flatten_dataset(dataset):
    flattened_examples = []

    for example in dataset:
        for paragraph in example['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                flattened_example = {
                    'question': qa['question'],
                    'context': context,
                    'answers': {
                        'text': [qa['answers'][0]['text']],
                        'answer_start': [qa['answers'][0]['answer_start']],
                    },
                    'title': 'BioASQ',
                    'id': qa['id']
                }
                flattened_examples.append(flattened_example)

    return flattened_examples

def preprocess_training_examples(examples, tokenizer, max_length = 384, stride = 128, model_checkpoint=None):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        
        sequence_ids = inputs.sequence_ids(i) if "xlnet" not in model_checkpoint else inputs.token_type_ids[i]

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_validation_examples(examples, tokenizer, max_length = 384, stride = 128, model_checkpoint=None):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i) if "xlnet" not in model_checkpoint else inputs.token_type_ids[i]
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

def load_bioasq_dataset(type='list'):
    local_bioasq_train_file = f'data/BioASQ/train/BioASQ-train-{type}.json'
    local_bioasq_val_file = f'data/BioASQ/val/BioASQ-val-{type}.json'
    local_bioasq_test_file = f'data/BioASQ/test/BioASQ-test-{type}.json'
    raw_dataset = load_dataset('json', data_files={'train': local_bioasq_train_file, 'val': local_bioasq_val_file, 'test': local_bioasq_test_file}, field='data')

    train_data = raw_dataset['train']
    val_data = raw_dataset['val']
    test_data = raw_dataset['test']

    train_examples = flatten_dataset(train_data)
    val_examples = flatten_dataset(val_data)
    test_examples = flatten_dataset(test_data)

    train_data = Dataset.from_dict({key: [example[key] for example in train_examples] for key in train_examples[0].keys()})
    val_data = Dataset.from_dict({key: [example[key] for example in val_examples] for key in val_examples[0].keys()})
    test_data = Dataset.from_dict({key: [example[key] for example in test_examples] for key in val_examples[0].keys()})

    raw_datasets = DatasetDict({
        'train': train_data,
        'validation': val_data,
        'test': test_data
    })
    return raw_datasets

def preprocess_datasets(tokenizer, raw_datasets, max_length, stride, model_checkpoint):
    train_dataset = raw_datasets["train"].map(
        partial(preprocess_training_examples, tokenizer=tokenizer, max_length=max_length, stride=stride, model_checkpoint=model_checkpoint),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].map(
        partial(preprocess_validation_examples, tokenizer=tokenizer, max_length=max_length, stride=stride, model_checkpoint=model_checkpoint),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    test_dataset = raw_datasets["test"].map(
        partial(preprocess_validation_examples, tokenizer=tokenizer, max_length=max_length, stride=stride, model_checkpoint=model_checkpoint),
        batched=True,
        remove_columns=raw_datasets["test"].column_names,
    )

    return train_dataset, validation_dataset, test_dataset


def create_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size):
    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")
    test_set = test_dataset.remove_columns(["example_id", "offset_mapping"])
    test_set.set_format("torch")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size,
    )
    eval_dataloader = DataLoader(
        validation_set, collate_fn=default_data_collator, batch_size=batch_size
    )
    test_dataloader = DataLoader(
        test_set, collate_fn=default_data_collator, batch_size=batch_size
    )

    return train_dataloader, eval_dataloader, test_dataloader