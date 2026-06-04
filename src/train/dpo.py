import argparse
import os
import torch
import json
import random
import transformers
from dotenv import load_dotenv

load_dotenv()
os.environ["WANDB_PROJECT"] = "CADFusion_VF"

from datasets import Dataset
from trl import DPOTrainer, DPOConfig
from utils import prepare_model_and_tokenizer


class CustomDPOTrainer(DPOTrainer):
    """
    Custom DPO Trainer that overrides get_batch_samples to match the Trainer signature.
    
    The base Trainer expects: get_batch_samples(epoch_iterator, num_batches, device)
    The DPOTrainer has: get_batch_samples(model, batch) for generation during eval
    
    This custom trainer adapts the signature to be compatible with the base Trainer.
    """
    
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        """
        Collect batch samples from the epoch iterator.
        
        This method matches the signature expected by the base Trainer class.
        It collects batches from the iterator and returns them along with the item count.
        
        Args:
            epoch_iterator: Iterator over batches
            num_batches: Number of batches to collect
            device: Device to put tensors on
            
        Returns:
            tuple: (batch_samples, num_items_in_batch)
        """
        batch_samples = []
        num_items_in_batch = None
        
        for _ in range(num_batches):
            try:
                batch_samples += [next(epoch_iterator)]
            except StopIteration:
                break

        if len(batch_samples) > 0 and "labels" in batch_samples[0]:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

        return batch_samples, num_items_in_batch
    
    def get_batch_samples_for_generation(self, model, batch):
        """
        Generate samples from the model and reference model for the given batch.
        
        This is the original DPOTrainer.get_batch_samples method, renamed to avoid
        signature conflicts with the base Trainer class.
        
        Args:
            model: The policy model
            batch: A single batch of data
            
        Returns:
            tuple: (policy_output_decoded, reference_output_decoded)
        """
        # Call the parent DPOTrainer's get_batch_samples method
        return super().get_batch_samples(model, batch)
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the DPO loss.
        
        This method matches the signature expected by the base Trainer class by accepting
        the num_items_in_batch parameter, even though DPO doesn't use it.
        
        Args:
            model: The model to compute loss for
            inputs: Input batch
            return_outputs: Whether to return outputs along with loss
            num_items_in_batch: Number of items in batch (unused by DPO, but required by Trainer signature)
            
        Returns:
            Loss tensor, or tuple of (loss, metrics) if return_outputs=True
        """
        # Call the parent DPOTrainer's compute_loss method, ignoring num_items_in_batch
        return super().compute_loss(model, inputs, return_outputs)
    
    def log(self, logs, start_time=None):
        """
        Log metrics on the various objects watching training.
        
        This method matches the signature expected by the base Trainer class by accepting
        the start_time parameter.
        
        Args:
            logs: Dictionary of metrics to log
            start_time: Optional start time for computing speed metrics
        """
        # Call the parent DPOTrainer's log method, which handles DPO-specific metric aggregation
        # The DPOTrainer.log internally calls super().log(logs), so start_time will be passed through
        return super().log(logs)

parser = argparse.ArgumentParser()
parser.add_argument("--run-name", type=str, required=True)
parser.add_argument("--lora-rank", type=int, default=32)
parser.add_argument("--lora-alpha", type=int, default=32)
parser.add_argument("--lora-dropout", type=float, default=0.05)
parser.add_argument("--sample-cutoff", default=100000, type=int)
parser.add_argument("--pretrained-path", type=str, required=True)
parser.add_argument("--data-path", type=str, required=True)
parser.add_argument("--output-path", type=str, required=True)
parser.add_argument("--num-epochs", type=int, default=3)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--eval-freq", default=1000, type=int)
parser.add_argument("--save-freq", default=500, type=int)
parser.add_argument("--debug", action="store_true", default=False)
args = parser.parse_args()



with open(args.data_path, 'r') as f:
    raw_data = json.load(f)
    
random.shuffle(raw_data)

if len(raw_data) > args.sample_cutoff + 100:
    ds = {
        "train": Dataset.from_list(raw_data[:args.sample_cutoff]),
        "val": Dataset.from_list(raw_data[-100:])
    }
else:
    ds = {
        "train": Dataset.from_list(raw_data[:-100]),
        "val": Dataset.from_list(raw_data[-100:])
        }

llama_model, llama_tokenizer = prepare_model_and_tokenizer(args)

for name, param in llama_model.named_parameters():
    if "lora" in name:  # Check if "lora" is in the parameter's name
        param.requires_grad = True
        
training_args = DPOConfig(
    run_name=args.run_name,
    learning_rate=1.41e-5, 
    per_device_train_batch_size=2,
    per_device_eval_batch_size=args.batch_size,
    report_to="wandb",
    num_train_epochs=args.num_epochs,
    do_eval=True,
    eval_steps=args.eval_freq,
    save_steps=args.save_freq,
    output_dir=args.output_path
    )

trainer = CustomDPOTrainer(
    llama_model,
    None,
    args=training_args,
    train_dataset=ds['train'],
    eval_dataset=ds['val'],
    tokenizer=llama_tokenizer,
)
trainer.save_model()
trainer.train()
trainer.save_model()