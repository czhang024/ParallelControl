import os
import fire
from datasets import load_dataset
from typing import Iterable, Union

glue_tasks = {"cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"}

# example usage: python download_glue.py --task_names mnli,qqp --data_dir ./data

def download_glue_tasks(task_names: Union[str, Iterable[str]], data_dir: str = "data"):
    print(f"downloading tasks: {str(task_names)}")
    
    if isinstance(task_names, str):
        task_names = [task_names]
    
    assert set(task_names).issubset(glue_tasks)
    
    for task_name in task_names:
        print(f"Downloading {task_name}...")
        save_dir = os.path.join(data_dir, "glue", task_name)
        os.makedirs(save_dir, exist_ok=True)
        
        try:
            # Try the new dataset path first
            raw_datasets = load_dataset("nyu-mll/glue", task_name)
            raw_datasets.save_to_disk(save_dir)
            print(f"Successfully downloaded {task_name}")
        except Exception as e:
            print(f"Error downloading {task_name}: {e}")
            # Try alternative approaches
            try:
                print(f"Trying alternative method for {task_name}...")
                raw_datasets = load_dataset("glue", task_name)
                raw_datasets.save_to_disk(save_dir)
                print(f"Successfully downloaded {task_name} using alternative method")
            except Exception as e2:
                print(f"Failed to download {task_name} with both methods: {e2}")

if __name__ == "__main__":
    fire.Fire(download_glue_tasks)