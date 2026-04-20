import os
import pickle
from os.path import join

import configargparse
import lightning as L
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from dataset_llm import CustomDataset
from model_llm import LLM, get_model_name
from util import get_base_dir

tqdm.pandas()
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# Helper function
def keep_first_five(values, row_examples):
    return ",".join(values.split(" [SEP] ")[:row_examples])


# Data loading
def load_data(train_csv_file, test_csv_file, validation_csv_file, row_examples, args):
    train_df = pd.read_csv(train_csv_file)
    test_df = pd.read_csv(test_csv_file)
    validation_df = pd.read_csv(validation_csv_file)
    train_df['data'] = train_df['data'].apply(keep_first_five, row_examples=row_examples)
    test_df['data'] = test_df['data'].apply(keep_first_five, row_examples=row_examples)
    validation_df['data'] = validation_df['data'].apply(keep_first_five, row_examples=row_examples)
    return train_df, test_df, validation_df


# DataLoader creation
def create_dataloaders(df, text_column, label_column, batch_size):
    L.seed_everything(seed=253)

    # Create train dataset and dataloader
    data_dataset = CustomDataset(
        texts=df[text_column].values.tolist(),
        targets=df[label_column].values.tolist(),
        table_id=df['table_id'].values.tolist(),
        column_id=df['col_idx'].values.tolist()
    )
    data_dataloader = DataLoader(
        dataset=data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=16, 
        pin_memory=True
    )
    return data_dataloader

def get_data_embedding(model, dataloader, tokenizer, device, data_type, args, task, base_dir):
    filename = (
        f"{task}_{args.data_name}_{data_type}_{args.row_examples}_"
        f"{args.model_name.split('/')[-1]}_{args.quantized}_{args.mean_pooling}_"
        f"1_none.pkl"
    )
    output_dir = join(base_dir, 'data', 'representation', args.data_name, task)
    os.makedirs(output_dir, exist_ok=True)
    filename = join(output_dir, filename)

    # Check if file already exists
    if False:  # os.path.exists(filename):
        print(f"Embedding already exist {filename}")
    else:
        print(f"Generating embeddings and saving to {filename}")
        total_embeddings = []
        total_labels = []
        total_table_ids = []
        total_column_ids = []
        model.eval()
        with torch.no_grad():
            for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
                prompt, targets, table_ids, col_ids = batch
                current_prompts = list(prompt)

                success = False
                truncate_ratio = 1

                while not success:
                    try:
                        # Truncate each prompt based on current ratio
                        trimmed_prompts = [
                            p[: int(len(p) * truncate_ratio)] for p in current_prompts
                        ]

                        # Tokenize and move to device
                        encodings = tokenize_text(trimmed_prompts, tokenizer)
                        input_ids = encodings["input_ids"].to(device)
                        attention_mask = encodings["attention_mask"].to(device)

                        # Get model output
                        representation = model(input_ids, attention_mask).cpu().numpy()

                        # Store output
                        total_embeddings.append(representation)
                        total_labels.extend(targets.cpu().numpy())
                        total_table_ids.extend(table_ids)
                        total_column_ids.extend(col_ids)

                        # Cleanup
                        del input_ids, attention_mask, representation
                        torch.cuda.empty_cache()
                        success = True

                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"⚠️ OOM at {int(truncate_ratio * 100)}% tokens. Retrying with shorter input...")
                            torch.cuda.empty_cache()
                            truncate_ratio *= 0.9
                        else:
                            raise e
                        
        # Convert lists to arrays
        total_embeddings = np.concatenate(total_embeddings, axis=0)
        total_labels = np.array(total_labels)
        total_table_ids = np.array(total_table_ids)
        total_column_ids = np.array(total_column_ids)

        # Save to a file
        data = {
            'table_id': total_table_ids,
            'column_id': total_column_ids,
            'embedding': total_embeddings,
            'label': total_labels
        }
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        print(f"Embeddings saved to {filename}")


def tokenize_text(text, tokenizer):
    """
    Tokenize the text and return PyTorch tensors with dynamic padding
    """
    encodings = tokenizer(
        text,
        return_tensors='pt',
        padding='longest',  # Dynamically pad each batch to the length of the longest sequence
        add_special_tokens=False
    )

    return encodings


def main():
    parser = configargparse.ArgParser()
    parser.add_argument("--gpu", type=str, default="cuda",
                        help="Which GPU to use")
    parser.add_argument('--row_examples', type=int, default=25,
                        help="batch size used for training, validation and test")
    parser.add_argument('--batch-size', type=int, default=1,
                        help="batch size used for training, validation and test")
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help="batch size used for training, validation and test")
    parser.add_argument('--data-name', type=str, default='dataset-sota',
                        help="for loading dataset")
    parser.add_argument('--model-name', type=str, default='qwen',
                        help="Name of the large language model (LLM) to use.")
    parser.add_argument("--quantized", action="store_true", default=True,
                        help="Whether to quantize the model or not")
    parser.add_argument("--mean-pooling", action="store_true", default=True,
                        help="Whether to use mean pooling for LLM embeddings")

    args = parser.parse_args()
    base_dir = get_base_dir()
    device = torch.device(args.gpu)
    model_name = get_model_name(args)

    task = 'cta'

    if args.data_name == 'dataset-sota-small':
        train_csv_file = "data/schema_sota_train_" + task + "_small.csv"
        test_csv_file = "data/schema_sota_test_" + task + ".csv"
        validation_csv_file = "data/schema_sota_validation_" + task + ".csv"
    elif args.data_name == 'dataset-sota':
        train_csv_file = "data/cpadata/schema/schema_sota_" + task + "_training.csv"
        test_csv_file = "data/cpadata/schema/schema_sota_" + task + "_test.csv"
        validation_csv_file = "data/cpadata/schema/schema_sota_" + task + "_validation.csv"
    elif args.data_name == 'dataset-sota-dbpedia':
        train_csv_file = "./data/sota_train_dbpedia.csv"
        test_csv_file = "./data/sota_test_dbpedia.csv"
        validation_csv_file = "./data/sota_validation_dbpedia.csv"
    elif args.data_name == 'dataset-t2d':
        train_csv_file = "data/train_t2d_doduo.csv"
        validation_csv_file = "data/train_t2d_doduo.csv"
        test_csv_file = "data/test_t2d_doduo.csv"
    elif args.data_name == 'dataset-turl':
        train_csv_file = "data/train_turl_doduo.csv"
        validation_csv_file = "data/validation_turl_doduo.csv"
        test_csv_file = "data/test_turl_doduo.csv"
    else:
        raise ValueError(f"Unknown data-name: {args.data_name}")

    train_csv_file = join(base_dir, train_csv_file)
    test_csv_file = join(base_dir, test_csv_file)
    validation_csv_file = join(base_dir, validation_csv_file)

    text_column = "data"
    label_column = "class_id"

    access_token = os.getenv("HF_TOKEN")
    if access_token:
        login(access_token)

    L.seed_everything(seed=253)
    model = LLM(args, access_token, args.gpu, args.mean_pooling)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_df, test_df, validation_df = load_data(
        train_csv_file, test_csv_file, validation_csv_file, args.row_examples, args
    )
    train_df[text_column] = tokenizer.bos_token + train_df[text_column]
    test_df[text_column] = tokenizer.bos_token + test_df[text_column]
    validation_df[text_column] = tokenizer.bos_token + validation_df[text_column]

    train_dataloader = create_dataloaders(
        train_df, text_column, label_column, args.batch_size * args.accumulation_steps
    )
    test_dataloader = create_dataloaders(
        test_df, text_column, label_column, args.batch_size * args.accumulation_steps
    )
    validation_dataloader = create_dataloaders(
        validation_df, text_column, label_column, args.batch_size * args.accumulation_steps
    )

    get_data_embedding(model, test_dataloader, tokenizer, device, 'test', args, task, base_dir)
    get_data_embedding(model, validation_dataloader, tokenizer, device, 'validation', args, task, base_dir)
    get_data_embedding(model, train_dataloader, tokenizer, device, 'train', args, task, base_dir)


if __name__ == "__main__":
    main()
