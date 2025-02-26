# TOFU Attack 

> NOTE: full details coming soon!

Create a new attack method in `methods/` called `<YOUR-ATTACK-NAME>.py`. Inside this file, define a method with the following signature:

```python 
def attack(hf_link, tokenizer, alt_candidates) -> dict:

    # Placeholder 
    return {
        0 : alt_candidates[0],
        1 : alt_candidates[1],
        2 : alt_candidates[2],
    } 

```

The method should return a dictionary of `int` rankings (0 : len(alt_candidates)) and the corresponding alt_candidate.

Then, run `python eval.py --attack <YOUR-ATTACK-NAME>` and the results will be stored in `results/<YOUR-ATTACK-NAME>.json`.

## Attack Methods

Attacks will be in `methods/` and should be formatted as `<output>_<num>_<input>` where `<output>` is what we are measuring (ex. loss, token spread) and `<input>` is what we are changing (ex. perturbing, quantizing). A full table of all combinations/attack methods will be added soon!

## TODOs

* (Advit) Map over remaining attacks (gradient-based)
* (Megan) Add-in mixup-based attacks (retain + forget combos)
* (Advit) Rewrite `eval.py` to only have `results` and then separate function to aggregate/parse
* (Megan) Add more attacks from literature