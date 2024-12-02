# generate_and_eval.py
from generate_system import TaskDecomposition
from eval_system import EvaluationConfig, EvaluationSystem
from llm_dev import LLM
import json
import os
from datetime import datetime
from pathlib import Path

class GenerateAndEval:
    def __init__(self, output_dir="output"):
        """
        Initialize all necessary components for generation and evaluation
        
        Args:
            output_dir: Base directory for outputs (default: "output")
        """
        self.llm = LLM(model_type='claude-sonnet', verbose=True)
        self.decomposer = TaskDecomposition(self.llm, output_language="English")
        
        self.config = EvaluationConfig(
            committee_llms=["claude-sonnet", "gpt-4o", "llama3-70b", "qwen-72b"],
            chief_llm="claude-sonnet",
            extractor_llm="claude-sonnet",
            verbose=True
        )
        
        self.eval_system = EvaluationSystem(self.config)
        self.base_output_dir = Path(output_dir)
        
    def _ensure_output_dir(self, task_type):
        """Create output directory if it doesn't exist"""
        output_dir = self.base_output_dir / task_type
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
        
    def process_single_task(self, task, reference_output="", task_type="default", dataset_idx=None):
        """
        Process a single task with generation and evaluation
        
        Args:
            task: Input task text
            reference_output: Reference output for comparison (default: "")
            task_type: Type of task for organizing outputs (default: "default")
            dataset_idx: Index of the data in the dataset (default: None)
            
        Returns:
            dict: Complete result including all outputs and evaluations
        """
        # Ensure output directory exists
        output_dir = self._ensure_output_dir(task_type)
        
        # Execute task decomposition
        result = self.decomposer.execute_task(task, enable_checking=True)
        multiple_step_result = result["final_result"]
        one_step_result = self.llm.generate_text(task)
        
        # Execute evaluation
        try:
            report_multiple_step = self.eval_system.evaluate(
                content=multiple_step_result, 
                task_info=task
            )
            report_one_step = self.eval_system.evaluate(
                content=one_step_result, 
                task_info=task
            )
            report_reference = self.eval_system.evaluate(
                content=reference_output, 
                task_info=task
            )
        except Exception as e:
            print(f"Error during evaluation: {e}")
            return None
            
        # Prepare complete result
        complete_result = {
            "dataset_index": dataset_idx,  # 添加数据集索引
            "reports": {
                "multiple_step": report_multiple_step,
                "one_step": report_one_step,
                "reference": report_reference
            },
            "outputs": {
                "multiple_step": multiple_step_result,
                "one_step": one_step_result,
                "reference_output": reference_output
            },
            "task": task,
            "execution_log": result["execution_log"]
        }
        
        return complete_result
    
    def save_result(self, result, start_time, series_id, task_type="default"):
        """
        Save result to file with appropriate naming
        
        Args:
            result: Result dict to save
            start_time: Start time for file naming
            series_id: Series ID for file naming
            task_type: Type of task (default: "default")
        """
        if result is None:
            return
            
        output_dir = self._ensure_output_dir(task_type)
        filename = f"{start_time}_{task_type}_{series_id}.json"
        
        with open(output_dir / filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            
        # Save summary for quick access
        summary = {
            "id": f"{start_time}_{series_id}",
            "dataset_index": result["dataset_index"],  # 添加数据集索引
            "task_type": task_type,
            "scores": {
                "multiple_step": result["reports"]["multiple_step"]["summary"]["overall_scores"],
                "one_step": result["reports"]["one_step"]["summary"]["overall_scores"],
                "reference": result["reports"]["reference"]["summary"]["overall_scores"]
            },
            "detailed_report": str(output_dir / filename)
        }
        
        summary_dir = output_dir / "summaries"
        summary_dir.mkdir(exist_ok=True)
        
        with open(summary_dir / f"{start_time}_{series_id}_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary