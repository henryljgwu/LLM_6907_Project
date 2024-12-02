# main.py
from datasets import load_dataset
from datetime import datetime
from generate_and_eval import GenerateAndEval
from tqdm import tqdm

def process_dataset(start_idx=0, num_samples=None, task_type="rewrite"):
    """
    Process dataset with specified number of samples
    
    Args:
        start_idx: Starting index for processing (default: 0)
        num_samples: Number of samples to process (default: None, process all)
        task_type: Type of task for organizing outputs (default: "rewrite")
    """
    # Load dataset
    ds = load_dataset("positivethoughts/rewrite_10k")
    
    # Initialize processor
    processor = GenerateAndEval()
    
    # Get current timestamp
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Determine end index
    if num_samples is None:
        end_idx = len(ds['train'])
    else:
        end_idx = min(start_idx + num_samples, len(ds['train']))
    
    # Process specified samples
    for idx in tqdm(range(start_idx, end_idx)):
        # Get example
        example = ds['train'][idx]
        task = example['prompt']
        reference = example['rewritten_text']
        
        print(f'-----\n{example}\n-----\n{task}\n-----\n{reference}\n-----\n')
        
        # Generate series ID
        series_id = f"{idx+1:04d}"
        
        # Process task
        result = processor.process_single_task(
            task=task,
            reference_output=reference,
            task_type=task_type,
            dataset_idx=idx  # 传入数据集索引
        )
        
        # Save result
        processor.save_result(result, start_time, series_id, task_type)
        
        # Print current progress with dataset index
        print(f"\nProcessed dataset index: {idx}")

if __name__ == "__main__":
    # Example usage: Process 5 - 25 samples
    process_dataset(start_idx=5, num_samples=20)