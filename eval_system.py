import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from llm_dev import LLM

@dataclass
class EvaluationConfig:
    committee_llms: List[str]
    chief_llm: str = "claude"
    extractor_llm: str = "claude"
    verbose: bool = False

class EvaluationSystem:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.chief = LLM(config.chief_llm, verbose=config.verbose)
        self.extractor = LLM(config.extractor_llm, verbose=config.verbose)
        self.committee = [LLM(llm_type, verbose=config.verbose) 
                         for llm_type in config.committee_llms]

    def _generate_standards_text(self, content: str, task_info: str) -> str:
        """Generate evaluation standards in text format"""
        prompt = f"""
        Please create comprehensive evaluation standards for the following content.
        
        Content to evaluate: {content}
        Task Information: {task_info}

        Requirements:
        1. Create 3-5 distinct evaluation aspects
        2. For each aspect, provide:
           - Clear name
           - Detailed description
           - Specific scoring criteria (what constitutes different score ranges)
           - Point allocation (out of 100 total points)
        3. Include a clear scoring method for calculating the final score
        4. Specify the exact output format evaluators should follow

        Example format:
        ASPECT 1: Content Quality (30 points)
        Description: Evaluates the accuracy, completeness, and relevance of the content
        Scoring Criteria:
        - 25-30 points: Exceptional quality, comprehensive coverage, no errors
        - 20-24 points: Good quality, mostly complete, minor errors
        - 15-19 points: Average quality, some gaps, several errors
        - Below 15: Poor quality, major gaps, significant errors

        [Continue with other aspects...]

        SCORING METHOD:
        Final score is calculated as the sum of all aspect scores, totaling 100 points.

        OUTPUT FORMAT:
        Evaluators should provide:
        1. Score and detailed justification for each aspect
        2. Any notable observations
        3. Final score with summary
        """
        return self.chief.generate_text(prompt)

    def _extract_standards_json(self, standards_text: str) -> Dict:
        """Extract JSON structure from text standards"""
        prompt = f"""
        Convert the following evaluation standards into a structured JSON format.
        
        Standards Text:
        {standards_text}

        Required JSON structure:
        {{
            "aspects": [
                {{
                    "name": "aspect name",
                    "description": "detailed description",
                    "scoring_criteria": {{
                        "range1": "criteria description",
                        "range2": "criteria description"
                    }},
                    "max_points": number
                }}
            ],
            "scoring_method": {{
                "description": "how final score is calculated",
                "total_points": 100
            }},
            "output_format": {{
                "required_fields": ["scores", "justifications", "observations"],
                "structure": {{
                    "aspect_scores": "object with aspect names as keys",
                    "justifications": "object with aspect names as keys",
                    "final_score": "number",
                    "summary": "string"
                }}
            }}
        }}

        Ensure the JSON maintains all the detailed criteria and scoring ranges from the original text.
        """
        return self.extractor.generate_json(prompt, {})

    def _extract_evaluation_report(self, llm_output: str, standards: Dict) -> Optional[Dict]:
        """Extract evaluation report into JSON format"""
        prompt = f"""
        Convert the following evaluation output into JSON format according to the specified structure.
        
        Evaluation Text:
        {llm_output}

        Required JSON structure:
        {{
            "aspect_scores": {{
                "aspect_name": number,  // for each aspect
            }},
            "justifications": {{
                "aspect_name": "detailed justification",  // for each aspect
            }},
            "final_score": number,
            "summary": "overall evaluation summary"
        }}

        Example JSON:
        {{
            "aspect_scores": {{
                "Content Quality": 27,
                "Technical Accuracy": 24
            }},
            "justifications": {{
                "Content Quality": "The content demonstrates exceptional thoroughness...",
                "Technical Accuracy": "The technical implementations are correct..."
            }},
            "final_score": 85,
            "summary": "Overall, this submission shows strong quality..."
        }}

        Ensure all scores and aspects from the evaluation text are accurately captured in the JSON.
        """
        return self.extractor.generate_json(prompt, {})

    def evaluate(self, content: str, task_info: str) -> Dict:
        """Execute complete evaluation process"""
        # 1. Generate and extract standards
        standards_text = self._generate_standards_text(content, task_info)
        standards = self._extract_standards_json(standards_text)
        if not standards:
            raise ValueError("Failed to generate evaluation standards")

        # 2. Collect committee evaluations
        committee_reports = []
        for llm in self.committee:
            eval_prompt = f"""
            Evaluate the following content according to these evaluation standards:
            
            Content to evaluate: {content}
            
            Evaluation Standards:
            {standards_text}

            Please provide:
            1. A score and detailed justification for each aspect
            2. Specific examples or quotes supporting your evaluation
            3. A final score and comprehensive summary

            Example response format:
            ASPECT: Content Quality
            Score: 27/30
            Justification: The content excels in [...specific details...] as evidenced by [...examples...]

            [Continue with other aspects...]

            FINAL SCORE: XX/100
            Summary: [Comprehensive evaluation summary...]
            """
            
            raw_report = llm.generate_text(eval_prompt)
            report = self._extract_evaluation_report(raw_report, standards)
            
            if report:
                committee_reports.append(report)

        # 3. Generate final report
        final_report = {
            "metadata": {
                "chief_llm": self.config.chief_llm,
                "committee_llms": self.config.committee_llms,
                "extractor_llm": self.config.extractor_llm
            },
            "standards_text": standards_text,
            "standards_json": standards,
            "committee_reports": committee_reports,
            "summary": self._generate_summary(committee_reports, standards)
        }

        return final_report

    def _generate_summary(self, reports: List[Dict], standards: Dict) -> Dict:
        """Generate evaluation summary"""
        prompt = f"""
        Create a comprehensive summary of the following evaluation reports.
        
        Reports:
        {json.dumps(reports, ensure_ascii=False)}
        
        Standards:
        {json.dumps(standards, ensure_ascii=False)}

        Generate a JSON summary with this structure:
        {{
            "overall_scores": {{
                "average": number,
                "min": number,
                "max": number,
                "std_dev": number
            }},
            "aspect_analysis": {{
                "aspect_name": {{
                    "average_score": number,
                    "key_points": ["point1", "point2"]
                }}
            }},
            "consensus_points": ["point1", "point2"],
            "variations": ["variation1", "variation2"],
            "recommendations": ["rec1", "rec2"]
        }}
        """
        return self.extractor.generate_json(prompt, {})

def save_report(report: Dict, filename: str):
    """Save evaluation report to file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

# Usage example
if __name__ == "__main__":
    # Configure evaluation system
    config = EvaluationConfig(
        committee_llms=["claude-sonnet", "gpt-4o-mini", "claude-haiku"],
        chief_llm="claude-sonnet",
        extractor_llm="gpt-4o",
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
        # save_report(report, "evaluation_report.json")
        print(json.dumps(report,indent=2,ensure_ascii=False))
        print("Evaluation completed, report saved")
    except Exception as e:
        print(f"Error during evaluation: {e}")