# API-shaping planning & working doc

This is a doc for thinking and working out the shape of the API. Nothing here is finalized. 

### Outer, non-technical user experience
For a minimally-technical user who just wants to run a multi-turn eval where they feed in instructions for the UserSimulator as part of the dataset. What data do you provide, and how?

- which user simulation method to use
- per-sample user simulation instructions (possibly different schema for different methods?)
- attached alongside sample, somehow? as part of dataset?

```python
# my_multi_turn_eval.py
from inspect_multiturn import converse, llm_user


@task
def multi_turn_eval():
	return Task(
		dataset=some_dataset, # show shape of dataset
		solver=converse(
			target=???, # isn't this passed in by the Inspect Framework?
			user=llm_user(),
			max_turns=10,
			first_turn='dataset'
		),
		scorer=model_graded_fact() # whatever, choose any scorer
		
	)
```

### Inner, technical user experience