from core.models import create_model
from collections import Counter

feedback_list= [
"the lessons are very clear and the examples make it easy to understand"
" i do not enjoy the course, but sometimes the instructions for assignments are confusing"
"the pace of the class is too fast and i am struggling"
"the online materials are okay "
"i am statisfied with the course overall"

]

classifier = build_feedback_classifier()

result =[]
for feedback in feedback_list:
    result = classifier.invoke()

result_dict = result.model_dump()
result.append(result_dict)

sentiments_counts = Counter(
    result["sentiment"]
    for result in result
)
print("\n" + "=" * 60)
print("FEEDBACK SUMMARY")
print("=" * 60)

print(f"positive:{}")

for number, result in
enumerate(sorted_results,
start=1):