from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
from llm_dev import LLM, BaseLLM
import os
from datetime import datetime
from eval_system import save_report, EvaluationConfig, EvaluationSystem

@dataclass
class ExecutionStep:
    """Enhanced execution step data class"""
    step_id: int
    description: str
    prompt: str
    expected_output: Optional[str] = None
    key_points: Optional[List[str]] = None
    constraints: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None
    result: Optional[str] = None
    check_result: Optional[Dict[str, Any]] = None

class TaskDecomposition:
    def __init__(self, llm: LLM, output_language: str = "English"):
        """
        Initialize TaskDecomposition with language control
        
        Args:
            llm (LLM): LLM instance
            output_language (str): Desired output language (e.g., "English", "Chinese", "Spanish")
        """
        self.llm: BaseLLM = llm
        self.output_language = output_language
        self.conversation_history: List[Dict[str, str]] = []
        self.additional_info: Optional[Dict[str, Any]] = None
        
    def _add_to_history(self, role: str, content: str):
        """Add conversation to history"""
        self.conversation_history.append({"role": role, "content": content})
        
    def generate_plan(self, task_description: str) -> List[ExecutionStep]:
        """Generate detailed task execution plan"""
        planning_prompt = f"""
        Please analyze the following task and create a detailed, step-by-step execution plan.
        Please provide all output in {self.output_language}.

        Task Description: {task_description}

        Generate a comprehensive execution plan in JSON format following these guidelines:

        1. Step Structure:
        {{
            "steps": [
                {{
                    "step_id": 1,
                    "description": "Detailed step description",
                    "prompt": "Specific execution instructions",
                    "expected_output": "Description of what this step should produce",
                    "key_points": ["Key elements to address", "Important aspects to include"],
                    "constraints": ["Any limitations or requirements to consider"],
                    "dependencies": ["References to previous steps if any, should be a int only, represent the step_id"]
                }}
            ]
        }}

        2. Requirements for Each Step:
        - Description should be specific and actionable
        - Prompt should provide clear guidance and context
        - Include all necessary details for execution
        - Consider dependencies on previous steps
        - Specify quality criteria and expectations

        3. Step Planning Considerations:
        - Break down complex tasks into manageable pieces
        - Ensure logical progression between steps
        - Include specific details and examples where relevant
        - Consider edge cases and potential challenges
        - Maintain focus on overall task objectives

        4. Content Guidelines:
        - Be specific rather than generic
        - Include measurable outcomes
        - Provide context for each step
        - Specify any required research or reference materials
        - Include quality checks and validation criteria

        Remember: Generate all content in {self.output_language}, but make sure the key in JSON stay in English to match each other.
        Please ensure the generated plan is detailed enough that each step can be executed without requiring additional clarification.
        """
        
        self._add_to_history("user", planning_prompt)
        
        plan_json = self.llm.generate_json(
            prompt=planning_prompt,
            schema={"steps": list}
        )
        
        if not plan_json or "steps" not in plan_json:
            raise ValueError("Failed to generate plan")
            
        self._add_to_history("assistant", json.dumps(plan_json, ensure_ascii=False, indent=2))
        
        steps = []
        for step in plan_json["steps"]:
            steps.append(ExecutionStep(
                step_id=step["step_id"],
                description=step["description"],
                prompt=step["prompt"],
                expected_output=step.get("expected_output"),
                key_points=step.get("key_points"),
                constraints=step.get("constraints"),
                dependencies=step.get("dependencies")
            ))
            
        return steps
        
    def execute_step(self, step: ExecutionStep, previous_results: List[str]) -> str:
        """Execute a single step with enhanced context and guidance"""
        dependencies_context = self._build_dependencies_context(step, previous_results)
        
        additional_context = ""
        if self.additional_info:
            additional_context = "\nAdditional Context and Notes:\n"
            for category, items in self.additional_info.items():
                if items:  # only add non-empty categories
                    additional_context += f"\n{category.replace('_', ' ').title()}:\n"
                    additional_context += self._format_list(items)
        
        context_prompt = f"""
        Task Execution Step {step.step_id}
        Please provide all output in {self.output_language}.

        Some context from the user instruction that you may refer:
        {additional_context}
        
        Step Description: {step.description}

        Previous Context:
        {dependencies_context}

        Expected Output: {step.expected_output}

        Key Points to Address:
        {self._format_list(step.key_points)}

        Constraints to Consider:
        {self._format_list(step.constraints)}

        Detailed Instructions:
        {step.prompt}

        Requirements for Execution:
        1. Address all key points explicitly
        2. Follow all specified constraints
        3. Maintain alignment with previous steps
        4. Ensure output format matches expectations
        5. Focus on quality and completeness

        Remember: Generate your response in {self.output_language}.
        Please execute this step and provide a detailed response that meets all requirements.
        """
        
        self._add_to_history("user", context_prompt)
        
        result = self.llm.generate_text(
            prompt=context_prompt,
            max_tokens=2000
        )
        
        self._add_to_history("assistant", result)
        return result

    def check_step_result(self, step: ExecutionStep, result: str, task_description: str) -> Dict[str, Any]:
        """Enhanced result validation with detailed feedback"""
        
        additional_context = ""
        if self.additional_info:
            additional_context = "\nAdditional Context and Notes:\n"
            for category, items in self.additional_info.items():
                if items:  # only add non-empty categories
                    additional_context += f"\n{category.replace('_', ' ').title()}:\n"
                    additional_context += self._format_list(items)
        
        check_prompt = f"""
        Please perform a comprehensive evaluation of the step execution result.
        Please provide all output in {self.output_language}.

        Original Task: {task_description}
        
        Some context from the user instruction that you may refer:
        {additional_context}

        Step Information:
        - Description: {step.description}
        - Expected Output: {step.expected_output}
        - Key Points: {self._format_list(step.key_points)}
        - Constraints: {self._format_list(step.constraints)}

        Execution Result:
        {result}

        Please evaluate the result based on the following criteria and return a detailed analysis in JSON format:

        {{
            "passed": boolean,
            "scores": {{
                "completeness": (0-10),  // Did it address all required points?
                "constraints_met": (0-10),  // Were all constraints followed?
                "quality": (0-10),  // Overall quality of the output
                "coherence": (0-10)  // Logical flow and connection with other steps
            }},
            "analysis": {{
                "strengths": ["list", "of", "strengths"],
                "weaknesses": ["list", "of", "weaknesses"],
                "missing_points": ["key points", "not addressed"],
                "violated_constraints": ["constraints", "not met"]
            }},
            "improvement_suggestions": ["specific", "actionable", "suggestions"],
            "overall_feedback": "Detailed explanation of the evaluation"
        }}

        Remember: Generate all analysis and feedback in {self.output_language}.
        Provide specific examples and references when discussing strengths or weaknesses.
        """
        
        self._add_to_history("user", check_prompt)
        
        check_result = self.llm.generate_json(
            prompt=check_prompt,
            schema={
                "passed": bool,
                "scores": dict,
                "analysis": dict,
                "improvement_suggestions": list,
                "overall_feedback": str
            }
        )
        
        self._add_to_history("assistant", json.dumps(check_result, ensure_ascii=False, indent=2))
        return check_result

    def _build_dependencies_context(self, step: ExecutionStep, previous_results: List[str]) -> str:
        """Build context based on step dependencies"""
        if not step.dependencies or not previous_results:
            return "No dependencies on previous steps."
            
        context = "Relevant context from previous steps:\n\n"
        for dep in step.dependencies:
            step_num = int(dep) - 1
            if 0 <= step_num < len(previous_results):
                context += f"From {dep}:\n{previous_results[step_num]}\n\n"
        return context

    def _format_list(self, items: Optional[List[str]]) -> str:
        """Format list items for prompt display"""
        if not items:
            return "None specified"
        return "\n".join(f"- {item}" for item in items)

    def execute_task(self, task_description: str, enable_checking: bool = True) -> Dict[str, Any]:
        """Execute complete task with enhanced monitoring and control"""
        self.conversation_history = []
        self.additional_info = self.extract_additional_info(task_description)
        
        steps = self.generate_plan(task_description)
        results = []
        execution_log = []
        
        for step in steps:
            result = self.execute_step(step, results)
            step.result = result
            
            if enable_checking:
                check_result = self.check_step_result(step, result, task_description)
                step.check_result = check_result
                
                if not check_result["passed"]:
                    retry_prompt = self._generate_retry_prompt(step, check_result)
                    result = self.llm.generate_text(
                        prompt=retry_prompt,
                        max_tokens=2000
                    )
                    step.result = result
                    step.check_result = self.check_step_result(step, result, task_description)
            
            results.append(result)
            execution_log.append({
                "step_id": step.step_id,
                "description": step.description,
                "expected_output": step.expected_output,
                "result": step.result,
                "check_result": step.check_result
            })
        
        final_result = self._generate_final_result(task_description, results, steps)
        
        return {
            "final_result": final_result,
            "execution_log": execution_log,
            "conversation_history": self.conversation_history
        }

    def extract_additional_info(self, task_description: str) -> Dict[str, Any]:
        """Extract additional information beyond task requirements from the original input"""
        extraction_prompt = f"""
        Please analyze the following input and extract the raw text and raw information that are not direct task requirements precisely. Output in JSON format.
        Please provide all output in {self.output_language}.

        Input: {task_description}

        Extract and categorize in the following format:
        {{
            "raw_text": ["Additional contextual information or background"],
            "raw_information": ["Global constraints or preferences mentioned"]
        }}

        Note: Only include information that is truly additional context, not the core task requirements.
        """
        
        additional_info = self.llm.generate_json(
            prompt=extraction_prompt,
            schema={
                "context_notes": list,
                "constraints_global": list,
                "reference_materials": list,
                "special_instructions": list,
                "other_info": list
            }
        )
        
        return additional_info
    
    def _generate_retry_prompt(self, step: ExecutionStep, check_result: Dict[str, Any]) -> str:
        """Generate detailed retry prompt based on validation feedback"""
        
        additional_context = ""
        if self.additional_info:
            additional_context = "\nAdditional Context and Notes:\n"
            for category, items in self.additional_info.items():
                if items:  # only add non-empty categories
                    additional_context += f"\n{category.replace('_', ' ').title()}:\n"
                    additional_context += self._format_list(items)
        
        return f"""
        The previous execution of step {step.step_id} requires improvement.
        Please provide all output in {self.output_language}.

        Some context from the user instruction that you may refer:
        {additional_context}
        
        Original Description: {step.description}
        Expected Output: {step.expected_output}

        Previous Result: {step.result}

        Evaluation Feedback:
        - Scores: {json.dumps(check_result['scores'], indent=2)}
        - Missing Points: {', '.join(check_result['analysis']['missing_points'])}
        - Violated Constraints: {', '.join(check_result['analysis']['violated_constraints'])}

        Improvement Requirements:
        {self._format_list(check_result['improvement_suggestions'])}

        Please revise the output addressing all identified issues while maintaining:
        1. Original task objectives
        2. Consistency with previous steps
        3. All specified constraints
        4. Required quality standards

        Remember: Generate the improved version in {self.output_language}.
        Provide an improved version that addresses all feedback points.
        """

    def _generate_final_result(self, task_description: str, results: List[str], steps: List[ExecutionStep]) -> str:
        """Generate final result with comprehensive context integration"""
        
        additional_context = ""
        if self.additional_info:
            additional_context = "\nAdditional Context and Notes:\n"
            for category, items in self.additional_info.items():
                if items:  # only add non-empty categories
                    additional_context += f"\n{category.replace('_', ' ').title()}:\n"
                    additional_context += self._format_list(items)
        
        final_integration_prompt = f"""
        Please create a comprehensive final result integrating all completed steps.
        Please provide all output in {self.output_language}.

        Original Task Description:
        {task_description}
        
        Some context from the user instruction that you may refer:
        {additional_context}

        Step Results Summary:
        {self._format_steps_summary(steps, results)}

        Requirements for Final Integration:
        1. Ensure perfect alignment with original task requirements
        2. Maintain logical flow and coherence across all components
        3. Address all key points from individual steps
        4. Resolve any inconsistencies between steps
        5. Provide a polished and professional final output

        Remember: Generate the final result in {self.output_language}.
        Create a cohesive final result that successfully achieves all original task objectives while maintaining the quality and detail level of individual steps.
        """

        self._add_to_history("user", final_integration_prompt)
        final_result = self.llm.generate_text(
            prompt=final_integration_prompt,
            max_tokens=3000
        )
        self._add_to_history("assistant", final_result)
        return final_result

    def _format_steps_summary(self, steps: List[ExecutionStep], results: List[str]) -> str:
        """Format detailed summary of all steps and their results"""
        summary = ""
        for step, result in zip(steps, results):
            summary += f"\nStep {step.step_id}: {step.description}\n"
            summary += f"Expected Output: {step.expected_output}\n"
            summary += f"Result: {result}\n"
            if step.check_result:
                summary += f"Quality Scores: {json.dumps(step.check_result.get('scores', {}), indent=2)}\n"
            summary += "-" * 80 + "\n"
        return summary