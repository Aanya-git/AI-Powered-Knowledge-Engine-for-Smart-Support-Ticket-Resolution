import pandas as pd
from classify import classify_ticket

# Load pilot dataset
df = pd.read_csv("data/pilot_dataset.csv")

results = []
for _, row in df.iterrows():
    text = row["text"]
    expected = row["expected_category"]
    predicted = classify_ticket(text)
    results.append({"text": text, "expected": expected, "predicted": predicted})

# Evaluate accuracy
correct = sum(1 for r in results if r["expected"] == r["predicted"])
accuracy = correct / len(results) * 100

print("\nValidation Results:")
for r in results:
    print(f"- {r['text']} → Expected: {r['expected']} | Predicted: {r['predicted']}")
print(f"\nOverall Accuracy: {accuracy:.2f}% on {len(results)} samples")
