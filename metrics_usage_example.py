from rouge import Rouge

rouge = Rouge()

hypotheses = ["It was a bright cold day in April, and the clocks were striking thirteen.",
              "The sky above the port was the color of television, tuned to a dead channel.",
              "It was love at first sight."]

references = ["It was a bright cold day in April.",
              "The sky above the port was the color of television.",
              "It was love at first sight."]

scores = rouge.get_scores(hypotheses, references, avg=True)
print(scores)
