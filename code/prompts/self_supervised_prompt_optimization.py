### Task Specification

from pydantic import BaseModel, Field, ValidationError
from typing import List, Literal
import json
import openai

openai.api_key = # ENTER API KEY HERE

# Input Schema

class InputSchema(BaseModel):
    county: str = Field(..., description="Name of the county")
    age: int = Field(..., description="Age of the person in years")
    income: float = Field(..., description="Annual income of the person in USD")
    home_value: float = Field(..., description="Market value of the home in USD")
    education: Literal['high_school', 'bachelor', 'master', 'phd'] = Field(..., description="Highest education level attained of the person")
    preparedness: Literal['low', 'medium', 'high'] = Field(..., description="Disaster preparedness level of the person")
    risk_response_possibility: Literal['unlikely', 'neutral', 'likely'] = Field(..., description="Probability of taking risk response strategies of the person")
    number_of_kids: int = Field(..., description="Number of kids in the household")
    employment_status: Literal['employed', 'unemployed', 'student', 'retired'] = Field(..., description="Employment status of the person")
    marriage_status: Literal['single', 'married', 'divorced', 'widowed'] = Field(..., description="Marital status of the person")
    flood_zone: Literal['A', 'A07', 'A10', 'A12', 'AE', 'AHB', 'AOB', 'VE', 'X'] = Field(..., description="Flood zone classification of the area")
    elevation_difficulty: Literal['low', 'medium', 'high'] = Field(..., description="Elevation difficulty level of the area")
    average_wind_speed: float = Field(..., description="Average wind speed in the area in mph")
    history_storm_count: int = Field(..., description="Number of historical storms in the area")
    average_flood_depth: float = Field(..., description="Average flood depth in the area in feet")
    house_year: int = Field(..., description="Year the house was built")
    ownership: Literal['own', 'rent'] = Field(..., description="Ownership status of the house")

# Output Schema

class OutputSchema(BaseModel):
    migration_probability: int = Field(..., ge = 1, le = 5, description="Likelihood of migration on a scale from 1 (very unlikely) to 5 (very likely)")
    migration_decision: Literal['yes', 'no'] = Field(..., description="Decision on whether the person will migrate or not")
    top_3_factors: List[str] = Field(..., description="Top 3 factors influencing the migration decision")



### Seed Prompt

import dspy

dspy.config(lm = dspy.OpenAI(model = "gpt-5.1", api_key = openai.api_key))

# Signature

class Signature(dspy.Signature):
    """Predict a Florida resident's migration response to hurricane risk."""

    input: InputSchema
    output: OutputSchema

# Basic Model

class BasicModule(dspy.Module):
    """A basic model to predict migration response to hurricane risk in Florida."""

    signature = Signature

    def compute(self, input: InputSchema) -> OutputSchema:
        prompt = f"""
        Given the following information about a Florida resident, predict their migration probability (1-5), migration decision (yes/no), and the top 3 factors influencing their decision.

        Resident Information:
        - County: {input.county}
        - Age: {input.age}
        - Income: {input.income}
        - Home Value: {input.home_value}
        - Education: {input.education}
        - Preparedness: {input.preparedness}
        - Risk Response Possibility: {input.risk_response_possibility}
        - Number of Kids: {input.number_of_kids}
        - Employment Status: {input.employment_status}
        - Marriage Status: {input.marriage_status}
        - Flood Zone: {input.flood_zone}
        - Elevation Difficulty: {input.elevation_difficulty}
        - Average Wind Speed: {input.average_wind_speed}
        - History Storm Count: {input.history_storm_count}
        - Average Flood Depth: {input.average_flood_depth}
        - House Year: {input.house_year}
        - Ownership: {input.ownership}

        Please provide your response in the following JSON format:
        {{
            "migration_probability": <int from 1 to 5>,
            "migration_decision": "<yes/no>",
            "top_3_factors": ["<factor1>", "<factor2>", "<factor3>"]
        }}
        """

        response = openai.ChatCompletion.create(
            model="gpt-5.1",
            messages=[
                {"role": "system", "content": "You are an expert in disaster risk management and human migration."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        result_text = response['choices'][0]['message']['content']

        try:
            result_json = json.loads(result_text)
            output = OutputSchema(**result_json)
            return output
        except (json.JSONDecodeError, ValidationError) as e:
            raise ValueError(f"Failed to parse model output: {e}")



### Self-supervised Optimization

import random

# LLM Prompt Mutation

def mutate_prompt(original_prompt):
    mutation_prompt = f"""
    Improve the following prompt. Make it:
    - more explicit
    - more structured
    - more deterministic
    - clearer in constraints

    Keep semantics identical. Here is the prompt:

    {original_prompt}
    """

    res = openai.ChatCompletion.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": mutation_prompt}],
    )
    return res.choices[0].message["content"]

# LLM Critic

def critic_prompt(prompt):
    critic_instruction = f"""
    You are a critic that evaluates prompts for predicting migration response to hurricane risk.
    Given the following prompt, rate it on a scale from 1 to 10 based on:
    - clarity
    - determinism
    - constraint obedience
    - avoidance of hallucination

    You MUST respond with ONLY a JSON:
    {{ "score": number }}
    """

    res = openai.ChatCompletion.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": critic_instruction},
            {"role": "user", "content": prompt},
        ],
    )
    try:
        return json.loads(res.choices[0].message["content"])["score"]
    except:
        return 0

# Optimization Loop

def optimization(initial_prompt, iterations=50, population=6):
    current_prompt = initial_prompt
    best_score = 0

    for iteration in range(iterations):
        candidates = [mutate_prompt(current_prompt) for _ in range(population)]
        scores = [critic_prompt(candidate) for candidate in candidates]

        max_score = max(scores)
        if max_score > best_score:
            best_score = max_score
            current_prompt = candidates[scores.index(max_score)]
            print(f"Iteration {iteration + 1}: New best score {best_score}")
        else:
            print(f"Iteration {iteration + 1}: No improvement")

    return current_prompt

initial_prompt = """
Given the following information about a Florida resident, predict their migration probability (1-5), migration decision (yes/no), and the top 3 factors influencing their decision.
"""

best_prompt = optimization(initial_prompt)
print(best_prompt)

# Synthetic Evaluation

def generate_test_cases(n=20):
    prompt = f"""
    Generate {n} synthetic test cases for Florida residents with the following fields:
    - county
    - age
    - income
    - home_value
    - education
    - preparedness
    - risk_response_possibility
    - number_of_kids
    - employment_status
    - marriage_status
    - flood_zone
    - elevation_difficulty
    - average_wind_speed
    - history_storm_count
    - average_flood_depth
    - house_year
    - ownership
    Provide the output in JSON array format.
    """

    res = openai.ChatCompletion.create(
        model="gpt-5.1",
        messages=[{"role": "user", "content": prompt}],
    )
    try:
        return json.loads(res.choices[0].message["content"])
    except json.JSONDecodeError:
        return []
    
def evaluate_model(prompt, tests):
    failures = 0
    for t in tests:
        query = f"""
        Here is the task prompt:
        {prompt}

        Now produce valid OUTPUT ONLY for this input:
        {json.dumps(t)}
        """

        res = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": query}],
        )
        output = res.choices[0].message["content"]

        try:
            OutputSchema.parse_raw(output)
        except:
            failures += 1

    return failures


tests = generate_test_cases(20)
failures = evaluate_model(best_prompt, tests)

print("Failures:", failures)

# Render Final JSON

final_json = {
    "best_prompt": best_prompt,
    "input_schema": InputSchema.schema(),
    "output_schema": OutputSchema.schema(),
    "test_failures": failures
}