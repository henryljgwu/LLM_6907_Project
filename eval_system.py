import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llm import LLM

@dataclass
class EvaluationConfig:
    committee_llms: List[str]  # List of committee LLMs
    chief_llm: str = "claude"  # Chief evaluator LLM
    extractor_llm: str = "claude"  # JSON extractor LLM
    verbose: bool = False

class EvaluationSystem:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.chief = LLM(config.chief_llm, verbose=config.verbose)
        self.extractor = LLM(config.extractor_llm, verbose=config.verbose)
        self.committee = [LLM(llm_type, verbose=config.verbose) 
                         for llm_type in config.committee_llms]
        
    def _generate_evaluation_standards(self, content: str, task_info: str) -> Dict:
        """Have Chief generate evaluation standards and aspects"""
        prompt = f"""
        Please create comprehensive evaluation standards for the following content:
        
        Content: {content}
        Task Information: {task_info}
        
        Generate evaluation standards in JSON format, including:
        1. List of evaluation aspects (aspects): Each aspect should include name, description, and scoring criteria
        2. Total scoring method (scoring_method): How the total 100 points are distributed
        3. Output format requirements (output_format): Expected evaluation output format
        
        The standards should be detailed enough to ensure consistent evaluation across different evaluators.
        """
        
        schema = {
            "aspects": list,
            "scoring_method": dict,
            "output_format": dict
        }
        
        return self.chief.generate_json(prompt, schema)

    def _extract_evaluation_report(self, llm_output: str, 
                                 standards: Dict) -> Optional[Dict]:
        """Extract evaluation report into JSON format using Extractor"""
        prompt = f"""
        Please convert the following evaluation output into JSON format:
        
        Evaluation Output:
        {llm_output}
        
        Ensure the extracted JSON adheres to the format defined in these evaluation standards:
        {json.dumps(standards['output_format'], ensure_ascii=False)}
        
        Be precise in extracting all scores and comments while maintaining the exact structure.
        """
        
        return self.extractor.generate_json(prompt, standards['output_format'])

    def _validate_evaluation(self, report: Dict, standards: Dict) -> bool:
        """Validate if the evaluation report covers all required aspects"""
        required_aspects = {aspect['name'] for aspect in standards['aspects']}
        report_aspects = set(report.keys())
        return required_aspects.issubset(report_aspects)

    def evaluate(self, content: str, task_info: str) -> Dict:
        """Execute complete evaluation process"""
        # 1. Generate evaluation standards
        standards = self._generate_evaluation_standards(content, task_info)
        if not standards:
            raise ValueError("Failed to generate evaluation standards")

        # 2. Collect committee evaluations
        committee_reports = []
        for llm in self.committee:
            eval_prompt = f"""
            Please evaluate the following content according to these standards:
            
            Content: {content}
            
            Evaluation Standards:
            {json.dumps(standards, ensure_ascii=False)}
            
            Requirements:
            1. Strictly follow the provided output format
            2. Evaluate each aspect thoroughly
            3. Provide clear rationale for each score
            4. Ensure all scores align with the scoring criteria
            """
            
            raw_report = llm.generate_text(eval_prompt)
            report = self._extract_evaluation_report(raw_report, standards)
            
            if report and self._validate_evaluation(report, standards):
                committee_reports.append(report)

        # 3. Generate final report
        final_report = {
            "metadata": {
                "chief_llm": self.config.chief_llm,
                "committee_llms": self.config.committee_llms,
                "extractor_llm": self.config.extractor_llm
            },
            "standards": standards,
            "committee_reports": committee_reports,
            "summary": self._generate_summary(committee_reports, standards)
        }

        return final_report

    def _generate_summary(self, reports: List[Dict], 
                         standards: Dict) -> Dict:
        """Generate evaluation summary"""
        summary_prompt = f"""
        Please generate a comprehensive summary based on the following evaluation reports:
        
        Evaluation Reports:
        {json.dumps(reports, ensure_ascii=False)}
        
        Standards:
        {json.dumps(standards, ensure_ascii=False)}
        
        Generate a JSON format summary including:
        1. Overall scores and their statistical analysis
        2. Analysis of each evaluation aspect
        3. Key findings and consensus points
        4. Notable disagreements or variations in assessments
        5. Recommendations based on the evaluations
        
        Ensure the summary provides clear insights while maintaining objectivity.
        """
        
        return self.extractor.generate_json(summary_prompt, {})

def save_report(report: Dict, filename: str):
    """Save evaluation report to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

# Usage example
if __name__ == "__main__":
    # Configure evaluation system
    config = EvaluationConfig(
        committee_llms=["claude", "gpt", "claude"],
        chief_llm="claude",
        extractor_llm="claude",
        verbose=True
    )
    
    # Initialize system
    system = EvaluationSystem(config)
    
    # Content to evaluate
    content = "Content to be evaluated"
    task_info = "Task-related information and requirements"
    
    # Execute evaluation
    try:
        report = system.evaluate(content, task_info)
        save_report(report, "evaluation_report.json")
        print("Evaluation completed, report saved")
    except Exception as e:
        print(f"Error during evaluation: {e}")